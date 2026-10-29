from __future__ import annotations

import http.client
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from supermimic_lg import workbench as wb


def wait_until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("process did not reach expected state")


def test_command_validation_and_argv(tmp_path):
    replay = tmp_path / "with spaces; literal.npz"
    replay.touch()
    command = wb.build_command(dict(mode="world_model", replay_file=str(replay)), tmp_path, tmp_path / "run")
    assert str(replay) in command
    assert command[:3] == [sys.executable, "-u", "-m"]
    assert command[3] == "supermimic_lg.scripts.train_world_model"
    for bad in ({"mode": "shell"}, {"mode": "smoke", "steps": -1},
                {"mode": "smoke", "device": "cpu;echo hi"},
                {"mode": "world_model", "replay_file": "missing.npz"}):
        with pytest.raises(ValueError):
            wb.build_command(bad, tmp_path, tmp_path / "run")


def test_online_supermimic_command_uses_bptt_runner(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/train.py").touch()
    motion = tmp_path / "motion.npz"
    motion.touch()
    command = wb.build_command(dict(mode="supermimic", mjlab_root=str(tmp_path), motion_file=str(motion),
                                    backend="mujoco", device="cpu"), tmp_path, tmp_path / "run")
    assert command[3] == "supermimic_lg.scripts.train_supermimic"
    assert "--output-dir" in command and "--policy-updates" in command
    assert command[command.index("--history") + 1] == "3"
    assert not any("ppo" in word.lower() for word in command)


def test_online_supermimic_rough_terrain_requires_mjlab_and_is_forwarded(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/train.py").touch()
    motion = tmp_path / "motion.npz"
    motion.touch()
    args = dict(mode="supermimic", mjlab_root=str(tmp_path), motion_file=str(motion),
                terrain="rough", backend="mjlab", device="cuda:0")
    command = wb.build_command(args, tmp_path, tmp_path / "run")
    assert command[command.index("--terrain") + 1] == "rough"
    with pytest.raises(ValueError, match="需要选择 mjlab GPU 后端"):
        wb.build_command({**args, "backend": "mujoco", "device": "cpu"}, tmp_path, tmp_path / "run")


def test_completion_log_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(wb, "build_command", lambda *_: [sys.executable, "-u", "-c", "print('hello training')"])
    manager = wb.TrainingManager(tmp_path)
    try:
        job = manager.start({"mode": "smoke"})
        wait_until(lambda: manager.status()["current"]["status"] == "completed")
        assert "hello training" in manager.log(job["id"])
        assert manager.status()["history"][0]["exit_code"] == 0
        for invalid in ("../secret", ".", ".."):
            with pytest.raises(ValueError):
                manager.log(invalid)
    finally:
        manager.close()


def test_stop_terminates_parent_and_worker(tmp_path, monkeypatch):
    heartbeat = tmp_path / "heartbeat.txt"
    child = "import time; from pathlib import Path; p=Path(" + repr(str(heartbeat)) + ");\nwhile True:\n p.write_text(str(time.time())); time.sleep(.05)"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); time.sleep(120)"
    monkeypatch.setattr(wb, "build_command", lambda *_: [sys.executable, "-u", "-c", parent])
    manager = wb.TrainingManager(tmp_path, stop_timeout=1)
    try:
        manager.start({"mode": "smoke"})
        wait_until(heartbeat.exists)
        with pytest.raises(ValueError, match="已有训练"):
            manager.start({"mode": "smoke"})
        manager.stop()
        wait_until(lambda: manager.status()["current"]["status"] == "stopped")
        stamp = heartbeat.read_text()
        time.sleep(0.3)
        assert heartbeat.read_text() == stamp, "worker survived stop"
    finally:
        manager.close()


def test_failed_job_preserves_exit_status(tmp_path, monkeypatch):
    monkeypatch.setattr(wb, "build_command", lambda *_: [sys.executable, "-c", "raise SystemExit(7)"])
    manager = wb.TrainingManager(tmp_path)
    try:
        manager.start({"mode": "smoke"})
        wait_until(lambda: manager.status()["current"]["status"] == "failed")
        assert manager.status()["current"]["exit_code"] == 7
    finally:
        manager.close()


def test_http_origin_guard_and_ui(tmp_path):
    from http.server import ThreadingHTTPServer
    manager = wb.TrainingManager(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), wb.make_handler(manager))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
    try:
        conn.request("GET", "/")
        response = conn.getresponse()
        assert response.status == 200
        assert "训练工作台" in response.read().decode()
        conn.request("POST", "/api/start", body='{"mode":"smoke"}', headers={"Content-Type": "application/json", "Origin": "https://evil.example"})
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        assert manager.status()["current"] is None
        conn.request("GET", "/api/status", headers={"Host": "evil.example"})
        response = conn.getresponse()
        assert response.status == 403
        response.read()
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        manager.close()

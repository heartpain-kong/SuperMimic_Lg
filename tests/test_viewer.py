from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib

import numpy as np
import pytest

from supermimic_lg.viewer import encode_png
from supermimic_lg.workbench import TrainingManager, ViewerManager, build_viewer_command


def wait_until(predicate, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError("process did not reach expected state")


def test_png_encoder_preserves_renderer_pixels():
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    encoded = encode_png(rgb)
    assert encoded[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, offset = {}, 8
    while offset < len(encoded):
        length, = struct.unpack(">I", encoded[offset:offset + 4])
        kind, payload = encoded[offset + 4:offset + 8], encoded[offset + 8:offset + 8 + length]
        chunks[kind] = payload
        offset += 12 + length
    assert struct.unpack(">II", chunks[b"IHDR"][:8]) == (4, 3)
    assert zlib.decompress(chunks[b"IDAT"]) == b"".join(b"\0" + row.tobytes() for row in rgb)


def test_viewer_command_reads_training_config(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint, motion, xml = run / "checkpoint.pt", tmp_path / "motion.npz", tmp_path / "robot.xml"
    for path in (checkpoint, motion, xml):
        path.touch()
    (run / "config.json").write_text(json.dumps({"args": {"motion_file": str(motion), "robot_xml": str(xml)}}))
    command = build_viewer_command({"checkpoint": str(checkpoint)}, tmp_path, tmp_path / "view")
    assert command[3] == "supermimic_lg.viewer"
    assert command[command.index("--motion-file") + 1] == str(motion)
    assert command[command.index("--robot-xml") + 1] == str(xml)
    with pytest.raises(ValueError):
        build_viewer_command({"checkpoint": str(checkpoint), "start_frame": -1}, tmp_path, tmp_path / "view")


def test_viewer_stop_does_not_stop_training_and_controls_are_validated(tmp_path):
    builder = lambda *_: [sys.executable, "-c", "import time; time.sleep(60)"]
    train, viewer = TrainingManager(tmp_path, command_builder=builder), ViewerManager(tmp_path)
    viewer.command_builder = builder
    try:
        train.start({"mode": "smoke"})
        job = viewer.start({})
        viewer.control({"paused": True, "speed": .5, "view": "top", "reset": True})
        control = json.loads((Path(job["run_dir"]) / "control.json").read_text())
        assert control == dict(paused=True, speed=.5, view="top", reset=1)
        for invalid in ({"speed": float("nan")}, {"speed": 100}, {"paused": "false"}, {"view": "bad"}):
            with pytest.raises(ValueError):
                viewer.control(invalid)
        viewer.stop()
        wait_until(lambda: viewer.status()["current"]["status"] == "stopped")
        assert train.status()["current"]["status"] == "running"
        with pytest.raises(ValueError):
            viewer.frame("../secret")
    finally:
        viewer.close()
        train.close()

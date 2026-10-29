"""Loopback-only training workbench; no web framework or GPU import required."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files


def integer(data, name, default, low, high):
    value = data.get(name, default)
    if isinstance(value, bool) or str(value).strip() != str(int(value)):
        raise ValueError(f"{name} 必须是整数")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} 必须在 {low}–{high} 之间")
    return value


def input_path(data, name, root, directory=False):
    raw = str(data.get(name, "")).strip()
    if not raw:
        raise ValueError(f"请填写 {name}")
    path = Path(raw).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"路径不存在或类型错误：{path}")
    return path


def build_command(data, root, run_dir):
    """Build argv from a fixed set of training entry points, never shell text."""
    mode = data.get("mode", "smoke")
    command = [sys.executable, "-u", "-m"]
    if mode in ("tracking", "supermimic"):
        mjlab = input_path(data, "mjlab_root", root, directory=True)
        candidates = (mjlab, mjlab / "unitree_rl_mjlab_lgdog",
                      mjlab / "unitree_rl_mjlab_lgdog" / "unitree_rl_mjlab_lgdog")
        if not any((p / "src").is_dir() and (p / "scripts/train.py").is_file() for p in candidates):
            raise ValueError("mjlab_root 必须包含 LgDog 的 src/ 和 scripts/train.py")
        motion = input_path(data, "motion_file", root)
        if mode == "supermimic":
            backend = data.get("backend", "mjlab")
            if backend not in ("mjlab", "mujoco"):
                raise ValueError("backend 必须是 mjlab 或 mujoco")
            terrain = data.get("terrain", "flat")
            if terrain not in ("flat", "rough"):
                raise ValueError("terrain 必须是 flat 或 rough")
            if terrain == "rough" and backend != "mjlab":
                raise ValueError("楼梯/斜坡/起伏地形需要选择 mjlab GPU 后端")
            if backend == "mjlab" and not device(data).startswith("cuda:"):
                raise ValueError("mjlab GPU 后端需要选择 CUDA 设备")
            command += ["supermimic_lg.scripts.train_supermimic", "--mjlab-root", str(mjlab),
                        "--motion-file", str(motion), "--backend", backend, "--device", device(data),
                        "--terrain", terrain, "--output-dir", str(run_dir)]
            for field, flag, default, high in (("envs", "num-envs", 64, 65536), ("iterations", "iterations", 20, 10000000),
                    ("batch_size", "batch-size", 128, 65536), ("horizon", "horizon", 8, 4096),
                    ("history", "history", 3, 128),
                    ("collect_steps", "collect-steps", 16, 65536), ("wm_updates", "wm-updates", 10, 10000),
                    ("policy_updates", "policy-updates", 2, 10000), ("warmup_iterations", "warmup-iterations", 5, 1000000)):
                command += ["--" + flag, str(integer(data, field, default, 1, high))]
            command += ["--seed", str(integer(data, "seed", 42, 0, 2147483647))]
            if str(data.get("resume", "")).strip():
                command += ["--resume", str(input_path(data, "resume", root))]
            return command
        command += ["supermimic_lg.scripts.train_lgdog", "--mjlab-root", str(mjlab),
                    "--motion-file", str(motion),
                    f"--env.scene.num-envs={integer(data, 'envs', 64, 1, 65536)}",
                    f"--agent.max-iterations={integer(data, 'iterations', 20, 1, 10000000)}",
                    f"--agent.seed={integer(data, 'seed', 42, 0, 2147483647)}",
                    "--agent.logger=tensorboard", f"--agent.run-name=workbench_{run_dir.name}"]
    elif mode == "world_model":
        replay = input_path(data, "replay_file", root)
        command += ["supermimic_lg.scripts.train_world_model", str(replay), "--output",
                    str(run_dir / "world_model.pt"), "--device", device(data),
                    "--steps", str(integer(data, "steps", 1000, 1, 10000000)),
                    "--batch-size", str(integer(data, "batch_size", 256, 1, 65536)),
                    "--horizon", str(integer(data, "horizon", 8, 1, 4096))]
    elif mode == "smoke":
        robot = data.get("robot", "lgdog1")
        if robot not in ("lgdog1", "go2"):
            raise ValueError("robot 必须是 lgdog1 或 go2")
        command += ["supermimic_lg.scripts.smoke", "--robot", robot, "--device", device(data),
                    "--steps", str(integer(data, "steps", 3, 1, 10000000))]
    else:
        raise ValueError("未知训练模式")
    return command


def device(data):
    value = str(data.get("device", "cpu"))
    if value != "cpu" and not (value.startswith("cuda:") and value[5:].isdigit()):
        raise ValueError("device 必须是 cpu 或 cuda:编号")
    return value


def command_display(command):
    if os.name == "nt":
        return "& " + " ".join("'" + value.replace("'", "''") + "'" for value in command)
    return shlex.join(command)


class TrainingManager:
    def __init__(self, root, stop_timeout=10, command_builder=None):
        self.root = Path(root).resolve()
        self.run_root = self.root / "logs" / "workbench"
        self.stop_timeout = stop_timeout
        self.lock = threading.RLock()
        self.proc = None
        self.job = None
        self.output = None
        self.closed = False
        self.cleanup_running = False
        self.command_builder = command_builder

    def _save(self):
        path = Path(self.job["run_dir"]) / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.job, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _refresh(self):
        if self.cleanup_running:
            return
        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            self.job.update(exit_code=code, ended_at=time.time(),
                            status="stopped" if self.job["status"] == "stopping" else
                            ("completed" if code == 0 else "failed"))
            self.proc = None
            self.output.close()
            self.output = None
            self._save()

    def start(self, data):
        with self.lock:
            self._refresh()
            if self.closed:
                raise ValueError("工作台正在关闭")
            if self.proc is not None:
                raise ValueError("已有训练在运行，请先停止当前训练")
            run_dir = self.run_root / (time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
            command = (self.command_builder or build_command)(data, self.root, run_dir)
            run_dir.mkdir(parents=True)
            log_path = run_dir / "train.log"
            self.output = log_path.open("wb")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # A source checkout also works without an editable reinstall.
            source = str(Path(__file__).resolve().parent.parent)
            env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
            kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            try:
                self.proc = subprocess.Popen(command, cwd=self.root, env=env,
                                             stdout=self.output, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, **kwargs)
            except Exception:
                self.output.close()
                self.output = None
                raise
            self.job = dict(id=run_dir.name, mode=data.get("mode", "smoke"), status="running",
                            pid=self.proc.pid, command=command, command_display=command_display(command), started_at=time.time(),
                            ended_at=None, exit_code=None, run_dir=str(run_dir), log_path=str(log_path))
            self._save()
            return dict(self.job)

    def stop(self):
        with self.lock:
            self._refresh()
            if self.proc is None or self.job["status"] == "stopping":
                return
            proc = self.proc
            self.cleanup_running = True
            self.job["status"] = "stopping"
            self._save()
            threading.Thread(target=self._stop_tree, args=(proc,), daemon=True).start()

    def _stop_tree(self, proc):
        try:
            self._terminate_tree(proc)
            if proc.poll() is None:
                raise RuntimeError("训练进程未退出，请再次停止或检查系统进程")
        except Exception as exc:
            with self.lock:
                self.job["stop_error"] = str(exc)
                if self.output:
                    self.output.write(("\n[workbench] stop failed: " + str(exc) + "\n").encode("utf-8"))
                    self.output.flush()
                if proc.poll() is None:
                    self.job["status"] = "running"
                self._save()
        finally:
            with self.lock:
                self.cleanup_running = False
                if self.proc is proc:
                    self._refresh()

    def _terminate_tree(self, proc):
        if os.name == "nt":
            # A consoleless Windows process cannot receive CTRL_C. taskkill /T
            # terminates its descendants too; Linux uses graceful SIGINT first.
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
            proc.wait(timeout=30)
        else:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=self.stop_timeout)
                    # The leader may exit before a worker. Probe the group.
                    os.killpg(proc.pid, 0)
                except (ProcessLookupError, PermissionError):
                    break
                except subprocess.TimeoutExpired:
                    pass

    def status(self):
        with self.lock:
            self._refresh()
            jobs = []
            for path in sorted(self.run_root.glob("*/job.json"), reverse=True)[:30]:
                try:
                    item = json.loads(path.read_text(encoding="utf-8"))
                    if item["status"] in ("running", "stopping") and (not self.job or item["id"] != self.job["id"]):
                        item["status"] = "unknown"
                    jobs.append(item)
                except (OSError, ValueError, KeyError):
                    continue
            return dict(current=dict(self.job) if self.job else None, history=jobs)

    def log(self, job_id):
        if not job_id or job_id in (".", "..") or Path(job_id).name != job_id or "/" in job_id or "\\" in job_id:
            raise ValueError("无效训练编号")
        path = self.run_root / job_id / "train.log"
        if not path.is_file():
            raise ValueError("找不到训练日志")
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            return stream.read().decode("utf-8", errors="replace")

    def close(self):
        with self.lock:
            self.closed = True
        self.stop()
        deadline = time.monotonic() + self.stop_timeout * 3 + 35
        while self.proc is not None and time.monotonic() < deadline:
            time.sleep(0.1)


def viewer_config(data, root):
    checkpoint = input_path(data, "checkpoint", root)
    config = {}
    path = checkpoint.parent / "config.json"
    if path.is_file() and path.stat().st_size < 1048576:
        try:
            config = json.loads(path.read_text(encoding="utf-8")).get("args", {})
        except (OSError, ValueError, AttributeError):
            pass
    if not isinstance(config, dict):
        config = {}
    merged = {key: data.get(key) or config.get(key) for key in ("motion_file", "mjlab_root", "robot_xml")}
    motion = input_path(merged, "motion_file", root)
    # Resolve the XML without importing torch/mjlab in the HTTP server.
    if merged.get("robot_xml"):
        xml = input_path(merged, "robot_xml", root)
    else:
        project = input_path(merged, "mjlab_root", root, directory=True)
        candidates = (project, project / "unitree_rl_mjlab_lgdog",
                      project / "unitree_rl_mjlab_lgdog" / "unitree_rl_mjlab_lgdog")
        xml = next((p / "src/assets/robots/lgdog1/xmls/lgdog1.xml" for p in candidates
                    if (p / "src/assets/robots/lgdog1/xmls/lgdog1.xml").is_file()), None)
        if xml is None:
            raise ValueError("找不到 LgDog XML，请填写正确的仓库路径或 robot_xml")
    return checkpoint, motion, xml


def build_viewer_command(data, root, run_dir):
    checkpoint, motion, xml = viewer_config(data, root)
    command = [sys.executable, "-u", "-m", "supermimic_lg.viewer", "--checkpoint", str(checkpoint),
               "--motion-file", str(motion), "--robot-xml", str(xml), "--output-dir", str(run_dir),
               "--start-frame", str(integer(data, "start_frame", 0, 0, 100000000))]
    if data.get("collision_only") is True:
        command.append("--collision-only")
    return command


class ViewerManager(TrainingManager):
    def __init__(self, root):
        super().__init__(root, command_builder=build_viewer_command)
        self.run_root = self.root / "logs" / "viewer"
        self.controls = dict(paused=False, speed=1., view="side", reset=0)

    def start(self, data):
        with self.lock:
            job = super().start(dict(data, mode="viewer"))
            self.controls = dict(paused=False, speed=1., view="side", reset=0)
            return job

    def control(self, data):
        with self.lock:
            self._refresh()
            if self.proc is None or self.job["status"] != "running":
                raise ValueError("请先开始策略观看")
            controls = dict(self.controls)
            if "paused" in data:
                if type(data["paused"]) is not bool:
                    raise ValueError("paused 必须是布尔值")
                controls["paused"] = data["paused"]
            if "speed" in data:
                speed = float(data["speed"])
                if not math.isfinite(speed) or not .1 <= speed <= 4.:
                    raise ValueError("播放速度必须为 0.1–4 倍")
                controls["speed"] = speed
            if "view" in data:
                if data["view"] not in ("side", "front", "top"):
                    raise ValueError("无效视角")
                controls["view"] = data["view"]
            if data.get("reset") is True:
                controls["reset"] += 1
            path = Path(self.job["run_dir"]) / "control.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(controls), encoding="utf-8")
            tmp.replace(path)
            self.controls = controls
            return controls

    def status(self):
        with self.lock:
            self._refresh()
            result = {"current": dict(self.job) if self.job else None}
            result["playback"] = None
            result["error_log"] = ""
            if self.job:
                path = Path(self.job["run_dir"]) / "state.json"
                try:
                    result["playback"] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
                if self.job["status"] == "failed":
                    result["error_log"] = self.log(self.job["id"])[-6000:]
            return result

    def frame(self, job_id):
        with self.lock:
            if not self.job or self.job["id"] != job_id:
                raise ValueError("观看编号不匹配")
            path = Path(self.job["run_dir"]) / "frame.png"
            if not path.is_file():
                raise ValueError("画面尚未生成")
            return path.read_bytes()

    def checkpoints(self):
        candidates = list((self.root / "logs").rglob("checkpoint.pt"))
        result = []
        for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
            item = {"path": str(path), "name": str(path.relative_to(self.root)), "modified_at": path.stat().st_mtime}
            try:
                config = json.loads((path.parent / "config.json").read_text(encoding="utf-8")).get("args", {})
                if isinstance(config, dict):
                    for key in ("motion_file", "robot_xml", "mjlab_root"):
                        item[key] = config.get(key) or ""
            except (OSError, ValueError, AttributeError):
                pass
            result.append(item)
        return result


def make_handler(manager, viewer=None):
    viewer = viewer or ViewerManager(manager.root)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, payload, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def allowed(self):
            host = self.headers.get("Host", "")
            port = self.server.server_address[1]
            if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self.reply(403, {"error": "仅允许本机或 SSH 隧道访问"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{host}":
                self.reply(403, {"error": "拒绝跨站请求"})
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            from urllib.parse import urlparse, parse_qs
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    self.reply(200, files("supermimic_lg.web").joinpath("index.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/status":
                    self.reply(200, dict(**manager.status(), python=sys.executable, root=str(manager.root), platform=sys.platform))
                elif url.path == "/api/log":
                    job_id = parse_qs(url.query).get("id", [""])[0]
                    self.reply(200, {"log": manager.log(job_id)})
                elif url.path == "/api/viewer/checkpoints":
                    self.reply(200, {"checkpoints": viewer.checkpoints()})
                elif url.path == "/api/viewer/status":
                    self.reply(200, viewer.status())
                elif url.path == "/api/viewer/frame":
                    job_id = parse_qs(url.query).get("id", [""])[0]
                    self.reply(200, viewer.frame(job_id), "image/png")
                else:
                    self.reply(404, {"error": "找不到接口"})
            except (ValueError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

        def do_POST(self):
            if not self.allowed():
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.reply(415, {"error": "只接受 JSON"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError("请求大小无效")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("参数必须是对象")
                if self.path == "/api/start":
                    self.reply(200, manager.start(data))
                elif self.path == "/api/stop":
                    manager.stop()
                    self.reply(200, manager.status())
                elif self.path == "/api/preview":
                    command = build_command(data, manager.root, manager.run_root / "PREVIEW")
                    self.reply(200, {"command": command, "command_display": command_display(command)})
                elif self.path == "/api/viewer/start":
                    self.reply(200, viewer.start(data))
                elif self.path == "/api/viewer/control":
                    self.reply(200, viewer.control(data))
                elif self.path == "/api/viewer/stop":
                    viewer.stop()
                    self.reply(200, viewer.status())
                else:
                    self.reply(404, {"error": "找不到接口"})
            except (ValueError, OSError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="SuperMimic 本地训练工作台")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="训练工作目录和日志目录的根路径")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    manager = TrainingManager(args.root)
    viewer = ViewerManager(args.root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(manager, viewer))
    server.daemon_threads = True
    def shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"SuperMimic 工作台：http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Python: {sys.executable}\n训练目录: {manager.root}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        viewer.close()
        manager.close()
        server.server_close()


if __name__ == "__main__":
    main()

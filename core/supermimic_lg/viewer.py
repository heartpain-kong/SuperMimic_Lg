"""Isolated policy playback and offscreen frames for the local workbench."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import signal
import struct
import sys
import time
import zlib


def atomic_write(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def write_json(path, data):
    atomic_write(path, json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def encode_png(rgb):
    """Encode renderer RGB without adding an imaging dependency."""
    import numpy as np
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("renderer must return [height,width,3] RGB")
    height, width, _ = rgb.shape
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    pixels = b"".join(b"\0" + row.tobytes() for row in rgb)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels, 3)) + chunk(b"IEND", b""))


class PolicyPlayback:
    """A deterministic CPU physics rollout, with checkpoint-defined input shape."""
    def __init__(self, checkpoint, motion_file, xml, visuals=True, start_frame=0):
        import torch
        from .config import lgdog1_spec
        from .online_data import MotionSequence
        from .policy import Policy, PolicyShape
        from .scripts.train_supermimic import file_hash
        from .simulator import MujocoSimulator
        torch.set_num_threads(1)
        # Release the training checkpoint file before deserialization. In
        # particular, Windows writers must be able to replace it atomically.
        payload = Path(checkpoint).read_bytes()
        ck = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        del payload
        if ck.get("algorithm") != "supermimic_bptt":
            raise ValueError("请选择在线 SuperMimic 的 checkpoint.pt；不支持 PPO 或单独世界模型文件")
        metadata = ck["metadata"]
        self.training_terrain = metadata.get("terrain", "flat")
        if metadata["motion_sha256"] != file_hash(motion_file) or metadata["xml_sha256"] != file_hash(xml):
            raise ValueError("参考动作或机器人 XML 与检查点不一致，请选择训练时使用的文件")
        self.shape = PolicyShape(**metadata["policy_shape"])
        if (self.shape.n_joints, self.shape.n_bodies) != (12, 13):
            raise ValueError("观看入口目前只支持 LgDog1 的 12 关节/13 body 策略")
        hidden = (int(metadata["network_size"]),) * 2
        self.policy = Policy(self.shape, hidden, hidden).eval()
        self.policy.load_state_dict(ck["policy"])
        self.motion = MotionSequence(motion_file, self.shape.goal_horizon, "cpu")
        if not 0 <= start_frame < self.motion.length - 1:
            raise ValueError("起始帧必须在动作片段范围内，并留出至少一个后续帧")
        self.sim = MujocoSimulator(xml, 1, "cpu", visuals=visuals)
        self.robot = lgdog1_spec()
        self.start_frame = start_frame
        self.steps, self.resets, self.pending_reset = 0, 0, False
        self.checkpoint_iteration = ck["iteration"]
        self.checkpoint = str(Path(checkpoint).resolve())
        self.reset()

    def reset(self):
        import torch
        from .online_data import observation
        self.frame = self.start_frame
        self.sim.reset(torch.tensor([0]), self.motion.states[self.frame:self.frame + 1])
        self.previous_action = torch.zeros(1, 12)
        first = observation(self.sim.state(), self.previous_action, self.sim.context)
        self.history = first[:, None].repeat(1, self.shape.history, 1)
        self.pending_reset = False

    def step(self):
        import torch
        from .online_data import advance_history
        from .state import quat_apply_inverse
        if self.pending_reset:
            self.reset()
            self.resets += 1
        with torch.inference_mode():
            action = self.policy(self.history, self.motion.goal(torch.tensor([self.frame])))
            state = self.sim.step(action)
            if not torch.isfinite(state).all():
                raise FloatingPointError("观看时仿真出现非有限状态，已停止回放")
            self.frame += 1
            self.steps += 1
            self.history = advance_history(state, action, self.sim.context, self.history)
            self.previous_action = action
            g = state.new_tensor([0., 0., -1.]).expand(1, 3)
            gravity = quat_apply_inverse(state[:, 3:7], g)
            ref = quat_apply_inverse(self.motion.quat[self.frame:self.frame + 1, 0], g)
            self.pending_reset = (abs(float(state[0, 2] - self.motion.pos[self.frame, 0, 2])) > self.robot.height_threshold
                                  or float((gravity - ref).norm()) > self.robot.tilt_threshold
                                  or self.frame >= self.motion.length - 1)

    def telemetry(self):
        return dict(checkpoint=self.checkpoint, checkpoint_iteration=self.checkpoint_iteration,
                    training_terrain=self.training_terrain, playback_terrain="flat",
                    history=self.shape.history, frame_dim=self.shape.obs_dim, frame=self.frame,
                    motion_frames=self.motion.length, steps=self.steps, resets=self.resets,
                    root_height=float(self.sim.state()[0, 2]), pending_reset=self.pending_reset)


def main():
    p = argparse.ArgumentParser(description="SuperMimic 策略实时观看进程")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--motion-file", type=Path, required=True)
    p.add_argument("--mjlab-root", type=Path, default=Path("../unitree_rl_mjlab_lgdog"))
    p.add_argument("--robot-xml", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--collision-only", action="store_true")
    args = p.parse_args()
    # Set before importing MuJoCo: Linux servers render through EGL without X11.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from .simulator import resolve_robot_xml
    args.output_dir.mkdir(parents=True, exist_ok=True)
    playback = PolicyPlayback(args.checkpoint, args.motion_file,
                              resolve_robot_xml(args.mjlab_root, args.robot_xml),
                              visuals=not args.collision_only, start_frame=args.start_frame)
    renderer = None
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        renderer = mujoco.Renderer(playback.sim.model, height=360, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 1.25
        options = mujoco.MjvOption()
        options.geomgroup[3] = bool(args.collision_only)
        control = dict(paused=False, speed=1., view="side", reset=0)
        last_reset, last_render, last_step, frame_id = 0, 0., time.monotonic(), 0
        last_signature = None
        print(json.dumps({"viewer": "ready", **playback.telemetry()}), flush=True)
        while not stopped:
            try:
                updated = json.loads((args.output_dir / "control.json").read_text(encoding="utf-8"))
                if isinstance(updated, dict):
                    control.update(updated)
            except (OSError, ValueError):
                pass
            reset = int(control.get("reset", 0))
            if reset != last_reset:
                playback.reset()
                playback.resets += 1
                last_reset = reset
                last_render = 0.
            now = time.monotonic()
            paused = bool(control.get("paused", False))
            speed = float(control.get("speed", 1.))
            if not .1 <= speed <= 4.:
                raise ValueError("invalid playback speed")
            if not paused and now - last_step >= .02 / speed:
                playback.step()
                last_step = now
            elif paused:
                last_step = now
            if now - last_render >= 1. / 15.:
                view = control.get("view", "side")
                camera.azimuth, camera.elevation = {"side": (90., -20.), "front": (180., -20.), "top": (90., -80.)}[view]
                signature = (playback.steps, last_reset, view)
                if signature != last_signature:
                    camera.lookat[:] = playback.sim.data[0].xipos[playback.sim.model.body("trunk").id]
                    renderer.update_scene(playback.sim.data[0], camera=camera, scene_option=options)
                    atomic_write(args.output_dir / "frame.png", encode_png(renderer.render()))
                    frame_id += 1
                    last_signature = signature
                write_json(args.output_dir / "state.json", dict(**playback.telemetry(), paused=paused,
                           speed=speed, view=view, frame_id=frame_id, updated_at=time.time()))
                last_render = now
            time.sleep(.002)
    finally:
        if renderer is not None:
            renderer.close()
        playback.sim.close()


if __name__ == "__main__":
    main()

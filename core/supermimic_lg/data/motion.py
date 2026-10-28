from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MotionClip:
    """Robot agnostic motion file used by the SuperMimic adapter."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray
    fps: float = 50.0
    name: str = "motion"

    @property
    def frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def n_joints(self) -> int:
        return int(self.joint_pos.shape[1])

    @property
    def n_bodies(self) -> int:
        return int(self.body_pos_w.shape[1])


def _array(data: dict[str, np.ndarray], key: str, ndim: int) -> np.ndarray:
    if key not in data:
        raise KeyError(f"motion file is missing '{key}'")
    x = np.asarray(data[key], dtype=np.float32)
    if x.ndim != ndim:
        raise ValueError(f"{key} must have {ndim} dimensions, got {x.shape}")
    return x


def validate_motion_clip(clip: MotionClip, expected_joints: int | None = None,
                        expected_bodies: int | None = None) -> None:
    fields = (clip.joint_pos, clip.joint_vel, clip.body_pos_w, clip.body_quat_w,
              clip.body_lin_vel_w, clip.body_ang_vel_w)
    if clip.frames < 2:
        raise ValueError("motion clip needs at least two frames")
    if any(a.shape[0] != clip.frames for a in fields):
        raise ValueError("all motion arrays must have the same frame count")
    if clip.joint_vel.shape != clip.joint_pos.shape:
        raise ValueError("joint_vel shape must equal joint_pos shape")
    if clip.body_pos_w.shape[-1] != 3 or clip.body_lin_vel_w.shape[-1] != 3 or clip.body_ang_vel_w.shape[-1] != 3:
        raise ValueError("body vector arrays must end in dimension 3")
    if clip.body_quat_w.shape[-1] != 4:
        raise ValueError("body_quat_w must use wxyz quaternions")
    if expected_joints is not None and clip.n_joints != expected_joints:
        raise ValueError(f"motion has {clip.n_joints} joints, expected {expected_joints}")
    if expected_bodies is not None and clip.n_bodies != expected_bodies:
        raise ValueError(f"motion has {clip.n_bodies} bodies, expected {expected_bodies}")
    if not np.isfinite(np.concatenate([a.reshape(-1) for a in fields])).all():
        raise ValueError("motion contains NaN or Inf")


def load_motion_npz(path: str | Path, expected_joints: int | None = None,
                    expected_bodies: int | None = None) -> MotionClip:
    path = Path(path)
    with np.load(path) as z:
        clip = MotionClip(
            joint_pos=_array(z, "joint_pos", 2), joint_vel=_array(z, "joint_vel", 2),
            body_pos_w=_array(z, "body_pos_w", 3), body_quat_w=_array(z, "body_quat_w", 3),
            body_lin_vel_w=_array(z, "body_lin_vel_w", 3), body_ang_vel_w=_array(z, "body_ang_vel_w", 3),
            fps=float(z["fps"]) if "fps" in z else 50.0, name=path.stem,
        )
    validate_motion_clip(clip, expected_joints, expected_bodies)
    return clip


def save_motion_npz(path: str | Path, clip: MotionClip) -> None:
    validate_motion_clip(clip)
    np.savez_compressed(path, joint_pos=clip.joint_pos, joint_vel=clip.joint_vel,
                        body_pos_w=clip.body_pos_w, body_quat_w=clip.body_quat_w,
                        body_lin_vel_w=clip.body_lin_vel_w, body_ang_vel_w=clip.body_ang_vel_w,
                        fps=np.asarray(clip.fps, dtype=np.float32))

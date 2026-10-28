from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import RobotSpec, lgdog1_spec
from ..data.motion import MotionClip, validate_motion_clip


LGDOG1 = lgdog1_spec()


@dataclass(frozen=True)
class MotionSchema:
    robot: RobotSpec
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    root_body_name: str
    anchor_body_name: str


def lgdog1_motion_schema() -> MotionSchema:
    return MotionSchema(LGDOG1, LGDOG1.joint_names, LGDOG1.tracked_body_names,
                        LGDOG1.root_body_name, LGDOG1.anchor_body_name)


def state_from_mjlab(robot_data, env_origins=None, default_joint_pos: np.ndarray | None = None) -> np.ndarray:
    """Extract the canonical state from an mjlab Entity.data object.

    The function is kept duck typed so importing the package does not require
    mjlab.  Arrays are returned in ``[B, 37]`` for LgDog1 (71 for G1).
    """

    def first(*names):
        for name in names:
            if hasattr(robot_data, name):
                return getattr(robot_data, name)
        raise AttributeError(f"robot data is missing all of {names}")

    qpos = first("root_link_pos_w", "root_pos_w")
    if env_origins is not None:
        qpos = qpos - env_origins
    quat = first("root_link_quat_w", "root_quat_w")
    q = robot_data.joint_pos
    v = first("root_link_lin_vel_w", "root_lin_vel_w")
    w = first("root_link_ang_vel_b", "root_ang_vel_b")
    qd = robot_data.joint_vel
    parts = [qpos, quat, q, v, w, qd]
    try:
        import torch
        return torch.cat(parts, dim=-1)
    except ImportError:
        return np.concatenate(parts, axis=-1)


def reference_from_mjlab_motion(motion, frame, device=None):
    """Create the reference tuple expected by the generic imagination code."""

    attrs = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    values = []
    for attr in attrs:
        x = getattr(motion, attr)[frame]
        if device is not None:
            x = x.to(device)
        values.append(x)
    root_index = LGDOG1.tracked_body_names.index(LGDOG1.root_body_name)
    root = values[0][..., root_index, :]
    return (*values, root)


def validate_lgdog1_motion(path: str | Path) -> MotionClip:
    return __import__("supermimic_lg.data.motion", fromlist=["load_motion_npz"]).load_motion_npz(
        path, expected_joints=LGDOG1.n_joints, expected_bodies=LGDOG1.n_bodies
    )

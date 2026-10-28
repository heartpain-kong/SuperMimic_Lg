from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .motion import MotionClip, save_motion_npz, validate_motion_clip


def convert_mjlab_npz(input_path: Path, output_path: Path, joint_order: list[int] | None = None,
                      body_order: list[int] | None = None, fps: float = 50.0) -> None:
    """Validate and reorder an existing mjlab motion file for a quadruped task."""

    with np.load(input_path) as z:
        joint_pos = np.asarray(z["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(z["joint_vel"], dtype=np.float32)
        body_pos = np.asarray(z["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(z["body_quat_w"], dtype=np.float32)
        body_lin = np.asarray(z["body_lin_vel_w"], dtype=np.float32)
        body_ang = np.asarray(z["body_ang_vel_w"], dtype=np.float32)
    if joint_order is not None:
        joint_pos, joint_vel = joint_pos[:, joint_order], joint_vel[:, joint_order]
    if body_order is not None:
        body_pos, body_quat = body_pos[:, body_order], body_quat[:, body_order]
        body_lin, body_ang = body_lin[:, body_order], body_ang[:, body_order]
    clip = MotionClip(joint_pos, joint_vel, body_pos, body_quat, body_lin, body_ang, fps=fps, name=input_path.stem)
    validate_motion_clip(clip)
    save_motion_npz(output_path, clip)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--joint-order", type=int, nargs="*")
    p.add_argument("--body-order", type=int, nargs="*")
    a = p.parse_args()
    convert_mjlab_npz(a.input, a.output, a.joint_order, a.body_order, a.fps)


if __name__ == "__main__":
    main()

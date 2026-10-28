from __future__ import annotations

import numpy as np


def test_lgdog_schema_is_simulator_ordered():
    from supermimic_lg.adapters.lgdog1 import LGDOG1

    assert LGDOG1.n_joints == 12
    assert LGDOG1.joint_names[:3] == ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint")
    assert LGDOG1.joint_names[-3:] == ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint")
    assert LGDOG1.tracked_body_names[0] == "trunk"
    assert LGDOG1.action_scale == 0.25


def test_motion_validation_and_roundtrip(tmp_path):
    from supermimic_lg.data.motion import MotionClip, load_motion_npz, save_motion_npz

    clip = MotionClip(
        np.zeros((3, 12), np.float32), np.zeros((3, 12), np.float32),
        np.zeros((3, 13, 3), np.float32), np.tile(np.array([1, 0, 0, 0], np.float32), (3, 13, 1)),
        np.zeros((3, 13, 3), np.float32), np.zeros((3, 13, 3), np.float32),
    )
    path = tmp_path / "lgdog.npz"
    save_motion_npz(path, clip)
    loaded = load_motion_npz(path, expected_joints=12, expected_bodies=13)
    assert loaded.joint_pos.shape == (3, 12)
    assert loaded.body_quat_w.shape == (3, 13, 4)


def test_state_canonical_is_yaw_invariant():
    import torch

    from supermimic_lg.state import QuadrupedStateSpec, quat_apply, quat_from_rotvec, quat_mul

    spec = QuadrupedStateSpec(12)
    state = torch.zeros(4, spec.state_dim)
    state[:, 3] = 1.0
    state[:, 7:19] = torch.randn(4, 12)
    state[:, 19:22] = torch.randn(4, 3)
    state[:, 22:25] = torch.randn(4, 3)
    state[:, 25:] = torch.randn(4, 12)
    yaw = torch.tensor([0.0, 0.5, -1.0, 2.0])
    yaw_q = quat_from_rotvec(torch.stack((torch.zeros(4), torch.zeros(4), yaw), dim=-1))
    rotated = state.clone()
    rotated[:, :2] += torch.randn(4, 2)
    rotated[:, 3:7] = quat_mul(yaw_q, state[:, 3:7])
    rotated[:, 19:22] = quat_apply(yaw_q, state[:, 19:22])
    torch.testing.assert_close(spec.canonical(state), spec.canonical(rotated), atol=1e-5, rtol=1e-5)

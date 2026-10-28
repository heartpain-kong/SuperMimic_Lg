from __future__ import annotations

import torch

from .state import quat_mul, quat_normalize


def _quat_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    q = quat_mul(quat_normalize(a) * a.new_tensor([1., -1., -1., -1.]), quat_normalize(b))
    return (2.0 * q[..., 1:]).square().sum(-1)


def tracking_reward(
    robot_pos: torch.Tensor,
    robot_quat: torch.Tensor,
    robot_lin_vel: torch.Tensor,
    robot_ang_vel: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_quat: torch.Tensor,
    ref_lin_vel: torch.Tensor,
    ref_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_limits: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    anchor_index: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable tracking reward used by imagined rollouts and adapters."""

    root_pos = (robot_pos[:, anchor_index] - ref_pos[:, anchor_index]).square().sum(-1)
    root_rot = _quat_error(robot_quat[:, anchor_index], ref_quat[:, anchor_index])
    body_pos = (robot_pos - ref_pos).square().mean(dim=(-1, -2))
    body_rot = _quat_error(robot_quat, ref_quat).mean(-1)
    lin = (robot_lin_vel - ref_lin_vel).square().mean(dim=(-1, -2))
    ang = (robot_ang_vel - ref_ang_vel).square().mean(dim=(-1, -2))
    lower, upper = joint_limits.unbind(-1)
    limit = (torch.relu(lower - joint_pos) + torch.relu(joint_pos - upper)).square().mean(-1)
    rate = (action - previous_action).square().mean(-1)
    terms = {
        "root_pos": torch.exp(-root_pos / 0.3**2),
        "root_rot": torch.exp(-root_rot / 0.4**2),
        "body_pos": torch.exp(-body_pos / 0.3**2),
        "body_rot": torch.exp(-body_rot / 0.4**2),
        "lin_vel": torch.exp(-lin / 1.0**2),
        "ang_vel": torch.exp(-ang / 3.14**2),
        "joint_limit": -2.0 * limit,
        "action_rate": -0.05 * rate,
    }
    total = sum(terms.values())
    return total, terms

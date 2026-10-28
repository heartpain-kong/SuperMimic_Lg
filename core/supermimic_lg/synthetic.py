from __future__ import annotations

import torch

from .config import RobotSpec
from .state import QuadrupedStateSpec, quat_normalize
from .training import ReplayWindow


def make_replay_window(robot: RobotSpec, batch_size: int = 32, horizon: int = 8,
                       context_dim: int | None = None, device: str = "cpu", seed: int = 0) -> ReplayWindow:
    """Generate a deterministic constant velocity replay for CPU smoke tests."""

    g = torch.Generator(device=device).manual_seed(seed)
    spec = QuadrupedStateSpec(robot.n_joints, robot.dt, device=device)
    context_dim = 4 + robot.n_joints if context_dim is None else context_dim
    state = torch.zeros(batch_size, spec.state_dim, device=device)
    state[:, 3] = 1.0
    state[:, 7:7 + robot.n_joints] = torch.randn(batch_size, robot.n_joints, generator=g, device=device) * 0.1
    state[:, 7 + robot.n_joints:10 + robot.n_joints] = 0.3
    actions = torch.randn(horizon, batch_size, robot.n_joints, generator=g, device=device) * 0.1
    contexts = torch.zeros(batch_size, context_dim, device=device)
    states = [state]
    done = []
    for t in range(horizon):
        next_state = spec.integrate(state, torch.cat((actions[t] * 0.02, torch.zeros(batch_size, spec.delta_dim - robot.n_joints, device=device)), dim=-1))
        states.append(next_state)
        done.append(torch.zeros(batch_size, device=device))
        state = next_state
    return ReplayWindow(torch.stack(states), actions, contexts, torch.stack(done), torch.stack(states[1:])[..., :3])

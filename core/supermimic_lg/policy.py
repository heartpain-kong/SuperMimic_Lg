from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .world_model import mlp


@dataclass
class PolicyShape:
    n_joints: int
    n_bodies: int
    history: int = 3  # Current frame plus two past frames.
    goal_horizon: int = 10
    latent_dim: int = 32

    @property
    def obs_dim(self) -> int:
        # q, qd, angular velocity, projected gravity and previous action.
        return 2 * self.n_joints + 3 + 3 + self.n_joints

    @property
    def goal_frame_dim(self) -> int:
        # q, qd, root gravity, root angular velocity, root linear velocity, height.
        return 2 * self.n_joints + 3 + 3 + 3 + 1

    @property
    def goal_dim(self) -> int:
        return self.goal_horizon * self.goal_frame_dim


class MotionEncoder(nn.Module):
    def __init__(self, shape: PolicyShape, hidden: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.net = mlp(shape.goal_dim, hidden, shape.latent_dim)

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(goal))


class Policy(nn.Module):
    """Goal encoder + history decoder with robot dependent action dimension."""

    def __init__(self, shape: PolicyShape, encoder_hidden=(512, 256), decoder_hidden=(512, 256)):
        super().__init__()
        self.shape = shape
        self.encoder = MotionEncoder(shape, encoder_hidden)
        self.decoder = mlp(shape.latent_dim + shape.history * shape.obs_dim, decoder_hidden, shape.n_joints)

    def forward(self, obs_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if obs_history.ndim == goal.ndim + 1:
            if obs_history.shape[-2:] != (self.shape.history, self.shape.obs_dim):
                raise ValueError("invalid frame-major observation history")
            obs_history = obs_history.flatten(-2)
        if obs_history.shape[-1] != self.shape.history * self.shape.obs_dim:
            raise ValueError("invalid flattened observation history")
        if goal.shape[-1] != self.shape.goal_dim:
            raise ValueError("invalid flattened goal")
        return torch.tanh(self.decoder(torch.cat((self.encoder(goal), obs_history), dim=-1)))


class Critic(nn.Module):
    def __init__(self, shape: PolicyShape, context_dim: int, hidden=(256, 128)):
        super().__init__()
        self.shape = shape
        self.net = mlp(shape.n_joints * 2 + 13 + shape.goal_dim + context_dim + 4, hidden, 1)

    def forward(self, canonical_state: torch.Tensor, goal: torch.Tensor, context: torch.Tensor,
                root_error: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((canonical_state, goal, context, root_error), dim=-1)).squeeze(-1)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..training import ReplayWindow


@dataclass
class ReplayStore:
    """Recorded simulator transitions for world-model fitting.

    NPZ schema: ``states[T+1,S]`` or ``states[T+1,N,S]``, ``actions[T,A]`` or
    ``actions[T,N,A]``, ``contexts[N,C]`` (or ``contexts[C]``), and optional
    ``done[T,N]``.  The collector should store the executed, noisy action.
    """

    states: torch.Tensor
    actions: torch.Tensor
    contexts: torch.Tensor
    done: torch.Tensor

    @classmethod
    def from_npz(cls, path: str | Path, device: str = "cpu") -> "ReplayStore":
        with np.load(path) as z:
            states = torch.as_tensor(z["states"], dtype=torch.float32, device=device)
            actions = torch.as_tensor(z["actions"], dtype=torch.float32, device=device)
            contexts = torch.as_tensor(z["contexts"], dtype=torch.float32, device=device)
            if "done" in z:
                done = torch.as_tensor(z["done"], dtype=torch.float32, device=device)
            else:
                done = torch.zeros(actions.shape[0], *(() if actions.ndim == 2 else (actions.shape[1],)), device=device)
        if states.ndim == 2:
            states, actions, done = states[:, None], actions[:, None], done[:, None]
        if contexts.ndim == 1:
            contexts = contexts[None]
        if contexts.shape[0] == 1 and states.shape[1] > 1:
            contexts = contexts.expand(states.shape[1], -1)
        if contexts.shape[0] != states.shape[1]:
            raise ValueError("contexts must have one row per replay environment")
        if states.shape[0] != actions.shape[0] + 1:
            raise ValueError("states must have one more frame than actions")
        if done.shape != actions.shape[:-1]:
            raise ValueError("done must have shape [T,N]")
        return cls(states, actions, contexts, done)

    @property
    def transitions(self) -> int:
        return int(self.actions.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[1])

    def sample(self, batch_size: int, horizon: int, generator: torch.Generator | None = None) -> ReplayWindow:
        if horizon > self.transitions:
            raise ValueError("horizon is longer than replay")
        starts_t = torch.randint(0, self.transitions - horizon + 1, (batch_size,), generator=generator, device=self.actions.device)
        envs = torch.randint(0, self.batch_size, (batch_size,), generator=generator, device=self.actions.device)
        idx_t = starts_t[None] + torch.arange(horizon + 1, device=self.actions.device)[:, None]
        states = self.states[idx_t, envs]
        a_t = starts_t[None] + torch.arange(horizon, device=self.actions.device)[:, None]
        actions = self.actions[a_t, envs]
        done = self.done[a_t, envs]
        contexts = self.contexts[envs]
        return ReplayWindow(states, actions, contexts, done, states[1:, ..., :3])

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import RobotSpec, WorldModelConfig
from .state import QuadrupedStateSpec, quat_normalize


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, zero_last: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    last = nn.Linear(d, out_dim)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


@dataclass
class WorldModelBatch:
    states: torch.Tensor
    actions: torch.Tensor
    contexts: torch.Tensor
    done: torch.Tensor


class WorldModel(nn.Module):
    """Differentiable dynamics plus a soft termination head.

    ``contexts`` is deliberately opaque to the model.  For LgDog it contains
    friction, encoder bias and COM offsets; other quadrupeds can add calibrated
    parameters without changing the state layout.
    """

    def __init__(self, robot: RobotSpec, context_dim: int | None = None,
                 cfg: WorldModelConfig | None = None, device: str | torch.device = "cpu"):
        super().__init__()
        self.robot = robot
        self.cfg = cfg or WorldModelConfig()
        self.spec = QuadrupedStateSpec(robot.n_joints, robot.dt, self.cfg.explicit_integration, device=device)
        # [friction, one encoder bias per joint, COM offset xyz].  A caller can
        # provide a larger context for additional calibration variables.
        self.context_dim = int(context_dim if context_dim is not None else 4 + robot.n_joints)
        self.action_dim = robot.n_joints
        self.action_scale = robot.action_scale
        self.dynamics = mlp(self.spec.state_dim + self.action_dim + self.context_dim,
                             self.cfg.hidden, self.spec.delta_dim, zero_last=True)
        self.termination = mlp(2, self.cfg.term_hidden, 1)
        self.register_buffer("context_mean", torch.zeros(self.context_dim))
        self.register_buffer("context_scale", torch.ones(self.context_dim))
        self.to(device)

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] != self.context_dim:
            raise ValueError(f"context dim {context.shape[-1]} != {self.context_dim}")
        return (context - self.context_mean) / self.context_scale.clamp_min(1e-6)

    def step(self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"action dim {action.shape[-1]} != {self.action_dim}")
        features = self.spec.canonical(state)
        delta = self.dynamics(torch.cat((features, action, self.normalize_context(context)), dim=-1))
        return self.spec.integrate(state, delta)

    def termination_logits(self, next_state: torch.Tensor, reference_root: torch.Tensor,
                           reference_quat: torch.Tensor | None = None) -> torch.Tensor:
        p = next_state[..., :3]
        q = quat_normalize(next_state[..., 3:7])
        h_error = reference_root[..., 2] - p[..., 2]
        # Difference between reference and predicted up vectors in their local z.
        g = next_state.new_tensor([0.0, 0.0, -1.0]).expand(q.shape[:-1] + (3,))
        qinv = q * q.new_tensor([1.0, -1.0, -1.0, -1.0])
        from .state import quat_apply
        tilt_pred = quat_apply(qinv, g)[..., 2]
        if reference_quat is None:
            tilt = tilt_pred + 1.0
        else:
            ref_q = quat_normalize(reference_quat)
            ref_qinv = ref_q * ref_q.new_tensor([1.0, -1.0, -1.0, -1.0])
            tilt = tilt_pred - quat_apply(ref_qinv, g)[..., 2]
        feats = torch.stack((h_error / self.robot.height_threshold, tilt / self.robot.tilt_threshold), dim=-1)
        return self.termination(feats).squeeze(-1)

    def window_loss(self, batch: WorldModelBatch, reference_root: torch.Tensor | None = None):
        if batch.states.shape[0] != batch.actions.shape[0] + 1:
            raise ValueError("states must have one more frame than actions")
        state = batch.states[0]
        loss_state = state.new_zeros(())
        loss_term = state.new_zeros(())
        term_acc = state.new_zeros(())
        k = batch.actions.shape[0]
        if reference_root is None:
            reference_root = batch.states[1][..., :3]
        for t in range(k):
            state = self.step(state, batch.actions[t], batch.contexts)
            loss_state = loss_state + self.spec.normalized_sq_error(state, batch.states[t + 1]).mean()
            logits = self.termination_logits(state, reference_root[t])
            target = batch.done[t].float()
            loss_term = loss_term + F.binary_cross_entropy_with_logits(logits, target)
            term_acc = term_acc + ((logits > 0) == (target > 0.5)).float().mean()
        loss_state, loss_term, term_acc = loss_state / k, loss_term / k, term_acc / k
        return loss_state + self.cfg.term_loss_weight * loss_term, {
            "state_loss": loss_state.detach(), "term_bce": loss_term.detach(), "term_acc": term_acc.detach()
        }

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, actions: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        state = states[0]
        errors = []
        for t in range(actions.shape[0]):
            state = self.step(state, actions[t], contexts)
            errors.append(self.spec.normalized_sq_error(state, states[t + 1]).mean())
        return torch.stack(errors)

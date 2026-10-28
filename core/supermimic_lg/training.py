from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .world_model import WorldModel, WorldModelBatch


@dataclass
class ReplayWindow:
    states: torch.Tensor  # [K+1, B, state_dim]
    actions: torch.Tensor  # [K, B, action_dim]
    contexts: torch.Tensor  # [B, context_dim]
    done: torch.Tensor  # [K, B]
    reference_root: torch.Tensor | None = None  # [K, B, 3]

    def as_batch(self) -> WorldModelBatch:
        return WorldModelBatch(self.states, self.actions, self.contexts, self.done)


class WorldModelTrainer:
    def __init__(self, model: WorldModel, lr: float = 3e-4, grad_clip: float = 1.0):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
        self.grad_clip = grad_clip

    def update(self, window: ReplayWindow) -> dict[str, float]:
        loss, metrics = self.model.window_loss(window.as_batch(), window.reference_root)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        out = {k: float(v.detach().cpu()) for k, v in metrics.items()}
        out.update(loss=float(loss.detach().cpu()), grad_norm=float(grad_norm.detach().cpu()))
        return out


def analytic_action_from_motion(joint_pos: torch.Tensor, previous_joint_pos: torch.Tensor,
                                default_joint_pos: torch.Tensor, scale: float) -> torch.Tensor:
    """Make a bounded PD target action for motion-only datasets.

    A simulator replay should use the actually executed action.  This helper is
    for bootstrapping a model from reference motion files when no controller log
    exists; the provenance is recorded by the CLI.
    """

    return ((joint_pos - default_joint_pos) / max(scale, 1e-6)).tanh()

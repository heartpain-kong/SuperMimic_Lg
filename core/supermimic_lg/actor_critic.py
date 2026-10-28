from __future__ import annotations

from dataclasses import dataclass

import torch

from .imagination import ImaginedRollout, ReferenceBatch, imagine
from .policy import Critic, Policy
from .world_model import WorldModel


def gae_targets(rewards: torch.Tensor, termination_probability: torch.Tensor,
                values: torch.Tensor, gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    """GAE targets on imagined transitions with soft termination."""

    out = torch.empty_like(rewards)
    running = values[-1]
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * (1.0 - termination_probability[t + 1]) * (
            (1.0 - lam) * values[t + 1] + lam * running
        )
        out[t] = running
    return out


@dataclass
class ActorCriticConfig:
    horizon: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_lr: float = 2e-4
    critic_lr: float = 2e-4
    critic_epochs: int = 2
    grad_clip: float = 1.0


class ImaginedActorCritic:
    """Short-horizon BPTT updates through a frozen world model."""

    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, cfg: ActorCriticConfig | None = None):
        self.wm, self.policy, self.critic = wm, policy, critic
        self.cfg = cfg or ActorCriticConfig()
        self.policy_opt = torch.optim.AdamW(policy.parameters(), lr=self.cfg.policy_lr, betas=(0.9, 0.95))
        self.critic_opt = torch.optim.AdamW(critic.parameters(), lr=self.cfg.critic_lr)

    @staticmethod
    def _root_error(states: torch.Tensor, roots: torch.Tensor) -> torch.Tensor:
        d = states[..., :3] - roots
        return torch.cat((d, states.new_zeros(d.shape[:-1] + (1,))), dim=-1)

    def update(self, state: torch.Tensor, context: torch.Tensor, obs_history: torch.Tensor,
               goals: torch.Tensor, refs: ReferenceBatch, previous_action: torch.Tensor,
               robot_bodies_fn, observation_fn=None) -> dict[str, float]:
        cfg = self.cfg
        self.wm.requires_grad_(False)
        rollout: ImaginedRollout = imagine(
            self.wm, self.policy, state, context, obs_history, goals, refs,
            cfg.horizon, previous_action, robot_bodies_fn, observation_fn,
        )
        root_error = self._root_error(rollout.states, refs.root_pos[: cfg.horizon + 1])
        canonical = torch.stack([self.wm.spec.canonical(s) for s in rollout.states])
        # Actor objective keeps the reward and terminal survival graph.  Critic
        # parameters are frozen during this pass, while its state input remains
        # differentiable so the bootstrap value can improve the policy through
        # the learned dynamics.
        self.critic.requires_grad_(False)
        values_actor = self.critic(
            canonical[-1], goals[cfg.horizon], context,
            root_error[-1],
        )
        discounts = cfg.gamma ** torch.arange(cfg.horizon, device=state.device, dtype=state.dtype)
        actor_objective = (discounts[:, None] * rollout.survival[:-1] * rollout.rewards).sum(0).mean()
        actor_objective = actor_objective + (cfg.gamma ** cfg.horizon) * rollout.survival[-1].mean() * values_actor.mean()
        actor_loss = -actor_objective
        with torch.no_grad():
            values = self.critic(canonical.reshape(-1, canonical.shape[-1]),
                                  goals[: cfg.horizon + 1].reshape(-1, goals.shape[-1]),
                                  context[None].expand(cfg.horizon + 1, -1, -1).reshape(-1, context.shape[-1]),
                                  root_error.reshape(-1, 4)).reshape(cfg.horizon + 1, -1)
        targets = gae_targets(rollout.rewards.detach(), rollout.termination_probability.detach(), values, cfg.gamma, cfg.gae_lambda)
        self.policy_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        policy_grad = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
        self.policy_opt.step()
        # Fit the critic to the same fixed imagined targets.
        self.critic.requires_grad_(True)
        for _ in range(cfg.critic_epochs):
            pred = self.critic(canonical[:-1].detach().reshape(-1, canonical.shape[-1]),
                               goals[: cfg.horizon].detach().reshape(-1, goals.shape[-1]),
                               context[None].expand(cfg.horizon, -1, -1).reshape(-1, context.shape[-1]),
                               root_error[:-1].detach().reshape(-1, 4))
            critic_loss = torch.nn.functional.mse_loss(pred, targets.detach().reshape(-1))
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
            self.critic_opt.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)
        return {
            "actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
            "imag_reward": float(rollout.rewards.mean().detach()),
            "survival_end": float(rollout.survival[-1].mean().detach()),
            "policy_grad_norm": float(policy_grad.detach()),
        }

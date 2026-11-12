"""Motion-tracking adapter to the shared BPTT objectives and optimization tools."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .imagination import ReferenceBatch, imagine
from .policy import Critic, Policy
from .world_model import WorldModel
from .learning.optim import frozen_parameters, optimizer_step
from .learning.returns import discounted_objective, lambda_returns


def gae_targets(rewards, termination_probability, values, gamma=.99, lam=.95):
    """Compatibility API: tracking supplies an unused initial done row [H+1,B]."""
    termination = termination_probability[1:] if len(termination_probability) == len(rewards)+1 else termination_probability
    return lambda_returns(rewards, termination, values, gamma, lam)


@dataclass
class ActorCriticConfig:
    horizon: int = 16
    gamma: float = .99
    gae_lambda: float = .95
    policy_lr: float = 2e-4
    critic_lr: float = 2e-4
    critic_epochs: int = 2
    critic_minibatches: int = 1
    grad_clip: float = 1.
    normalize_actor_loss: bool = False
    actor_beta2: float = .95
    actor_grad_value: float = 0.
    reward_limit: float = 0.
    value_target_limit: float = 0.


class ImaginedActorCritic:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, cfg=None):
        self.wm, self.policy, self.critic = wm, policy, critic
        self.cfg = cfg or ActorCriticConfig()
        self.policy_opt = torch.optim.AdamW(policy.parameters(), lr=self.cfg.policy_lr, betas=(.9, self.cfg.actor_beta2))
        self.critic_opt = torch.optim.AdamW(critic.parameters(), lr=self.cfg.critic_lr)

    @staticmethod
    def _root_error(states, roots):
        delta = states[..., :3]-roots
        return torch.cat((delta, states.new_zeros(delta.shape[:-1]+(1,))), -1)

    def update(self, state, context, obs_history, goals, refs: ReferenceBatch,
               previous_action, robot_bodies_fn, observation_fn=None):
        cfg = self.cfg
        with frozen_parameters(self.wm, self.critic):
            rollout = imagine(self.wm, self.policy, state, context, obs_history, goals, refs,
                              cfg.horizon, previous_action, robot_bodies_fn, observation_fn)
            root_error = self._root_error(rollout.states, refs.root_pos[:cfg.horizon+1])
            canonical = self.wm.spec.canonical(rollout.states.flatten(0, 1)).reshape(cfg.horizon+1, len(state), -1)
            bootstrap = self.critic(canonical[-1], goals[cfg.horizon], context, root_error[-1])
            rewards = rollout.rewards.clamp(-cfg.reward_limit, cfg.reward_limit) if cfg.reward_limit else rollout.rewards
            objective = discounted_objective(rewards, rollout.termination_probability[1:], bootstrap, cfg.gamma).mean()
            actor_loss = -objective/(cfg.horizon if cfg.normalize_actor_loss else 1.)
            with torch.no_grad():
                values = self.critic(canonical.flatten(0, 1), goals[:cfg.horizon+1].flatten(0, 1),
                    context.expand(cfg.horizon+1, -1, -1).flatten(0, 1), root_error.flatten(0, 1)).reshape(cfg.horizon+1, -1)
                targets = gae_targets(rewards, rollout.termination_probability, values,
                                      cfg.gamma, cfg.gae_lambda).flatten()
                if cfg.value_target_limit:
                    targets = targets.clamp(-cfg.value_target_limit, cfg.value_target_limit)
            grad = optimizer_step(actor_loss, self.policy, self.policy_opt, cfg.grad_clip, cfg.actor_grad_value)
        features = canonical[:-1].detach().flatten(0, 1)
        goals_x = goals[:cfg.horizon].detach().flatten(0, 1)
        contexts = context.detach().expand(cfg.horizon, -1, -1).flatten(0, 1)
        roots = root_error[:-1].detach().flatten(0, 1)
        for _ in range(cfg.critic_epochs):
            # A single minibatch preserves the original tracking update behavior.
            order = torch.arange(len(targets), device=state.device) if cfg.critic_minibatches == 1 else torch.randperm(len(targets), device=state.device)
            for index in order.tensor_split(min(cfg.critic_minibatches, len(targets))):
                prediction = self.critic(features[index], goals_x[index], contexts[index], roots[index])
                critic_loss = torch.nn.functional.mse_loss(prediction, targets[index])
                optimizer_step(critic_loss, self.critic, self.critic_opt, cfg.grad_clip)
        return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                "imag_reward": float(rollout.rewards.mean().detach()),
                "survival_end": float(rollout.survival[-1].mean().detach()), "policy_grad_norm": float(grad)}

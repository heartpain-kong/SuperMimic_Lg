"""Differentiable model rollout with an injected task; no simulator or file IO."""
from dataclasses import dataclass
from typing import Any, Protocol
import torch
from .returns import survival_weights


class DifferentiableTask(Protocol):
    def advance(self, condition: Any) -> Any: ...
    def observation(self, state, action, condition): ...
    def reward(self, state, action, previous_action, condition, context): ...
    def termination(self, world_model, state, context): ...
    def critic_features(self, world_model, states, context, conditions): ...


@dataclass
class ModelRollout:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    termination: torch.Tensor
    survival: torch.Tensor
    action_bounds: torch.Tensor
    conditions: tuple
    cached_features: tuple | None = None


def imagine(world_model, policy, task: DifferentiableTask, state, context, history,
            previous_action, condition, horizon, action_limit=1., reward_limit=10.):
    if horizon < 1:
        raise ValueError("imagination horizon must be positive")
    states, actions, rewards, termination, bounds, conditions = [state], [], [], [], [], [condition]
    for _ in range(horizon):
        raw = policy(history)
        action = raw.clamp(-action_limit, action_limit)
        nxt = world_model.step(state, action, context)
        r = task.reward(nxt, action, previous_action, condition, context)
        rewards.append(r.clamp(-reward_limit, reward_limit) if reward_limit else r)
        termination.append(task.termination(world_model, nxt, context))
        bounds.append((raw-action).square().mean(-1))
        condition = task.advance(condition)
        obs = task.observation(nxt, action, condition)
        history = torch.cat((history[:, 1:], obs[:, None]), 1)
        state, previous_action = nxt, action
        states.append(state)
        actions.append(action)
        conditions.append(condition)
    done = torch.stack(termination)
    return ModelRollout(torch.stack(states), torch.stack(actions), torch.stack(rewards), done,
                        survival_weights(done), torch.stack(bounds), tuple(conditions))

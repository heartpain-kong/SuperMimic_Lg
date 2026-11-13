"""Hoist reward/termination geometry out of the sequential policy recurrence.

Execution organization follows vendor/xmimic/xmimic/fast.py: normalize each new
observation once, batch rewards over H*B, and reuse physical critic features.
The LgDog MDP functions, integration rules and policy interface stay unchanged.
"""
import torch
from ...learning.rollout import ModelRollout
from ...learning.returns import survival_weights
from .task import VelocityCondition


class CompiledRecurrentStep:
    """Static-shape tensor step, following CuboiLeo's ImagStep organization.

    Compile only the recurrent step, not optimizer mutation or the whole H-step
    Python driver. Inductor is optional; aot_eager is useful for graph tests.
    """
    def __init__(self, world_model, policy, task, action_limit=1., backend='inductor'):
        self.wm, self.policy, self.task, self.action_limit = world_model, policy, task, action_limit
        self.fn = torch.compile(self.step, backend=backend, dynamic=False, fullgraph=True)

    def step(self, state, history, command, phase, context, depth):
        physical = self.wm.features(state, context)
        raw = self.policy.forward_normalized(history)
        action = raw.clamp(-self.action_limit, self.action_limit)
        state = self.wm.step_from_features(state, action, physical)
        condition = self.task.advance(VelocityCondition(command, phase, depth))
        obs = self.policy.obs_norm.apply(self.task.observation(state, action, condition))
        history = torch.cat((history[:, 1:], obs[:, None]), 1)
        return state, history, raw, action, physical, condition.phase

    def __call__(self, *args):
        return self.fn(*args)


def imagine_batched(world_model, policy, task, state, context, history,
                    previous_action, condition, horizon, action_limit=1., reward_limit=10., step_fn=None):
    if horizon < 1:
        raise ValueError('imagination horizon must be positive')
    batch = len(state)
    states, actions, raw_actions, conditions, features = [state], [], [], [condition], []
    history = policy.obs_norm.apply(history)
    for _ in range(horizon):
        if step_fn is None:
            physical = world_model.features(state, context)
            raw = policy.forward_normalized(history)
            action = raw.clamp(-action_limit, action_limit)
            state = world_model.step_from_features(state, action, physical)
            condition = task.advance(condition)
            obs = policy.obs_norm.apply(task.observation(state, action, condition))
            history = torch.cat((history[:, 1:], obs[:, None]), 1)
        else:
            state, history, raw, action, physical, phase = step_fn(
                state, history, condition.command, condition.phase, context, condition.depth)
            condition = VelocityCondition(condition.command, phase, condition.depth)
        features.append(physical)
        states.append(state)
        actions.append(action)
        raw_actions.append(raw)
        conditions.append(condition)
    features.append(world_model.features(state, context))
    states, actions = torch.stack(states), torch.stack(actions)
    previous = torch.cat((previous_action[None], actions[:-1]), 0)
    contexts = context[None].expand(horizon, -1, -1).reshape(horizon*batch, -1)
    reward_condition = VelocityCondition(
        torch.stack([c.command for c in conditions[:-1]]).flatten(0, 1),
        torch.stack([c.phase for c in conditions[:-1]]).flatten(0, 1),
        None if condition.depth is None else torch.stack([c.depth for c in conditions[:-1]]).flatten(0, 1))
    # Neither reward nor done feeds the recurrence. One large differentiable FK
    # call replaces H small calls; state/action temporal alignment is identical.
    next_states = states[1:].flatten(0, 1)
    rewards = task.reward(next_states, actions.flatten(0, 1), previous.flatten(0, 1),
                          reward_condition, contexts).reshape(horizon, batch)
    if reward_limit:
        rewards = rewards.clamp(-reward_limit, reward_limit)
    done = task.termination(world_model, next_states, contexts).reshape(horizon, batch)
    bounds = (torch.stack(raw_actions)-actions).square().mean(-1)
    return ModelRollout(states, actions, rewards, done, survival_weights(done), bounds,
                        tuple(conditions), tuple(features))


def critic_features(rollout, task, world_model, context, last_only=False):
    """Use cached per-state features without redoing height scans/canonicalization."""
    if rollout.cached_features is None:
        states = rollout.states[-1:] if last_only else rollout.states
        conditions = rollout.conditions[-1:] if last_only else rollout.conditions
        return task.critic_features(world_model, states, context, conditions)
    features = rollout.cached_features[-1:] if last_only else rollout.cached_features
    conditions = rollout.conditions[-1:] if last_only else rollout.conditions
    return torch.cat((torch.stack(features), torch.stack([c.command for c in conditions]),
                      torch.stack([c.phase for c in conditions])), -1)

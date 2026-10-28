from __future__ import annotations

from dataclasses import dataclass

import torch

from .policy import Policy
from .rewards import tracking_reward
from .world_model import WorldModel


@dataclass
class ReferenceBatch:
    """Reference values for H+1 frames, each already in the same body order."""

    pos: torch.Tensor
    quat: torch.Tensor
    lin_vel: torch.Tensor
    ang_vel: torch.Tensor
    root_pos: torch.Tensor


@dataclass
class ImaginedRollout:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    termination_probability: torch.Tensor
    survival: torch.Tensor


def imagine(wm: WorldModel, policy: Policy, state: torch.Tensor, context: torch.Tensor,
            obs_history: torch.Tensor, goals: torch.Tensor, refs: ReferenceBatch,
            horizon: int, previous_action: torch.Tensor, robot_bodies_fn,
            observation_fn=None) -> ImaginedRollout:
    """Roll out H steps and keep the graph for analytic policy gradients."""

    states, actions, rewards, dhat, survival = [state], [], [], [state.new_zeros(state.shape[:-1])], [state.new_ones(state.shape[:-1])]
    obs_hist = obs_history
    a_prev = previous_action
    alive = state.new_ones(state.shape[:-1])
    for t in range(horizon):
        action = policy(obs_hist, goals[t])
        next_state = wm.step(state, action, context)
        bodies = robot_bodies_fn(next_state)
        reward, _ = tracking_reward(
            *bodies, refs.pos[t + 1], refs.quat[t + 1], refs.lin_vel[t + 1], refs.ang_vel[t + 1],
            next_state[..., 7:7 + wm.robot.n_joints],
            next_state.new_tensor(wm.robot.joint_pos_limits), action, a_prev, 0
        )
        d = torch.sigmoid(wm.termination_logits(
            next_state, refs.root_pos[t + 1], refs.quat[t + 1][..., 0, :]
        ))
        alive = alive * (1.0 - d)
        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        dhat.append(d)
        survival.append(alive)
        if observation_fn is not None:
            obs_hist = observation_fn(next_state, action, context, obs_hist)
        state, a_prev = next_state, action
    return ImaginedRollout(torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(dhat), torch.stack(survival))

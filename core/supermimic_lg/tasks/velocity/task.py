"""LgDog velocity implementation of the differentiable task protocol."""
from dataclasses import dataclass
import math
import torch
from ...velocity_objective import objective_from
from .mdp import frame, reward, soft_termination


@dataclass(frozen=True)
class VelocityCondition:
    command: torch.Tensor
    phase: torch.Tensor
    depth: torch.Tensor | None = None


class VelocityTask:
    def __init__(self, feet, objective=None, dt=.02, gait_period=.6):
        self.feet, self.objective = feet, objective_from(objective)
        angle = 2*math.pi*dt/gait_period
        self.sin_step, self.cos_step = math.sin(angle), math.cos(angle)

    def advance(self, condition):
        sine, cosine = condition.phase.unbind(-1)
        phase = torch.stack((sine*self.cos_step+cosine*self.sin_step,
                             cosine*self.cos_step-sine*self.sin_step), -1)
        phase = phase*(condition.command.norm(dim=-1) >= 1e-3)[:, None]
        return VelocityCondition(condition.command, phase, condition.depth)

    def observation(self, state, action, condition):
        return frame(state, action, condition.command, condition.phase, condition.depth)

    def reward(self, state, action, previous_action, condition, context):
        return reward(state, action, previous_action, condition.command, condition.phase,
                      context, self.feet, self.objective)

    def termination(self, world_model, state, context):
        return soft_termination(state, context)

    def critic_features(self, world_model, states, context, conditions):
        """Time-major states -> time-major critic inputs, using one feature batch."""
        horizon, batch, _ = states.shape
        contexts = context.expand(horizon, -1, -1).reshape(-1, context.shape[-1])
        commands = torch.stack([c.command for c in conditions]).flatten(0, 1)
        phases = torch.stack([c.phase for c in conditions]).flatten(0, 1)
        physical = world_model.features(states.flatten(0, 1), contexts)
        return torch.cat((physical, commands, phases), -1).reshape(horizon, batch, -1)

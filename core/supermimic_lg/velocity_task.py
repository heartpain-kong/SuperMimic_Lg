"""Compatibility imports; implementations live in tasks.velocity."""
from .tasks.velocity.schema import STATE_DIM, CONTEXT_DIM, FRAME_DIM, context_dim, frame_dim
from .tasks.velocity.mdp import (sample_ground, ground_clearance, geometry, frame, phase_at,
    failure, velocity_error, state_features, FootKinematics, reward)
from .tasks.velocity.networks import VelocityPolicy, VelocityCritic, VelocityWorldModel
from .tasks.velocity.replay import VelocityReplay

__all__ = ["STATE_DIM", "CONTEXT_DIM", "FRAME_DIM", "context_dim", "frame_dim",
    "sample_ground", "ground_clearance", "geometry", "frame", "phase_at", "failure",
    "velocity_error", "state_features", "FootKinematics", "reward", "VelocityPolicy",
    "VelocityCritic", "VelocityWorldModel", "VelocityReplay"]

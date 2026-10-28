"""SuperMimic world-model training for generic quadrupeds and LgDog1.

The package keeps the differentiable model independent from mjlab.  The optional
``adapters`` modules translate simulator state and reference motion into the
canonical tensors used by the world model.
"""

from .config import RobotSpec, SuperMimicConfig, lgdog1_spec, unitree_go2_spec
from .state import CanonicalState, QuadrupedStateSpec
from .world_model import WorldModel
from .actor_critic import ActorCriticConfig, ImaginedActorCritic

__all__ = ["RobotSpec", "SuperMimicConfig", "lgdog1_spec", "unitree_go2_spec",
           "CanonicalState", "QuadrupedStateSpec", "WorldModel",
           "ActorCriticConfig", "ImaginedActorCritic"]

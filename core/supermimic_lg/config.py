from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobotSpec:
    """Robot dependent dimensions and physical conventions.

    ``joint_names`` follows the simulator order.  ``tracked_body_names`` is the
    reference body order used by the tracking reward.  The world model itself
    only needs the joint count; body kinematics are supplied by the adapter.
    """

    name: str
    joint_names: tuple[str, ...]
    tracked_body_names: tuple[str, ...]
    anchor_body_name: str
    root_body_name: str
    default_joint_pos: tuple[float, ...]
    joint_pos_limits: tuple[tuple[float, float], ...]
    action_scale: float
    dt: float = 0.02
    height_threshold: float = 0.20
    tilt_threshold: float = 0.80

    def __post_init__(self) -> None:
        n = len(self.joint_names)
        if n == 0:
            raise ValueError("joint_names cannot be empty")
        if len(self.default_joint_pos) != n:
            raise ValueError("default_joint_pos must match joint_names")
        if len(self.joint_pos_limits) != n:
            raise ValueError("joint_pos_limits must match joint_names")
        if self.anchor_body_name not in self.tracked_body_names:
            raise ValueError("anchor_body_name must be tracked")
        if self.root_body_name not in self.tracked_body_names:
            raise ValueError("root_body_name must be tracked")

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def n_bodies(self) -> int:
        return len(self.tracked_body_names)


@dataclass
class WorldModelConfig:
    hidden: tuple[int, ...] = (512, 512, 512)
    term_hidden: tuple[int, ...] = (64, 64)
    term_loss_weight: float = 1.0
    explicit_integration: bool = True


@dataclass
class PolicyConfig:
    history: int = 10
    goal_horizon: int = 10
    encoder_hidden: tuple[int, ...] = (512, 256)
    decoder_hidden: tuple[int, ...] = (512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    latent_dim: int = 32
    obs_noise: bool = True


@dataclass
class SuperMimicConfig:
    robot: RobotSpec
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    horizon: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95


def lgdog1_spec() -> RobotSpec:
    """Canonical LgDog1 mapping from the existing mjlab XML.

    The XML documents the order as ``[FR, FL, RR, RL] x [hip, thigh, calf]``.
    Body names intentionally use the existing mjlab names so the adapter can be
    used without renaming the asset.
    """

    legs = ("FR", "FL", "RR", "RL")
    joints = tuple(f"{leg}_{kind}_joint" for leg in legs for kind in ("hip", "thigh", "calf"))
    limits = tuple(((-1.2, 1.2), (-1.5, 1.5), (-0.85, 0.85))[i % 3] for i in range(12))
    bodies = ("trunk",) + tuple(f"{leg}_{kind}" for leg in legs for kind in ("hip", "thigh", "calf"))
    return RobotSpec(
        name="lgdog1",
        joint_names=joints,
        tracked_body_names=bodies,
        anchor_body_name="trunk",
        root_body_name="trunk",
        default_joint_pos=tuple(v for _ in legs for v in (0.0, 0.0, 0.2)),
        joint_pos_limits=limits,
        action_scale=0.25,
        dt=0.02,
    )


def unitree_go2_spec() -> RobotSpec:
    """A standard 12-DoF quadruped mapping for smoke tests and Go2 assets."""

    legs = ("FL", "FR", "RL", "RR")
    joints = tuple(f"{leg}_{kind}_joint" for leg in legs for kind in ("hip", "thigh", "calf"))
    bodies = ("base",) + tuple(f"{leg}_{kind}" for leg in legs for kind in ("hip", "thigh", "calf"))
    return RobotSpec(
        name="unitree_go2",
        joint_names=joints,
        tracked_body_names=bodies,
        anchor_body_name="base",
        root_body_name="base",
        default_joint_pos=tuple(v for _ in legs for v in (0.0, 0.8, -1.5)),
        joint_pos_limits=tuple(((-1.0, 1.0), (-1.8, 1.8), (-2.8, -0.1))[i % 3] for i in range(12)),
        action_scale=0.25,
        dt=0.02,
    )

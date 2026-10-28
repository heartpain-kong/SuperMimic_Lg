from __future__ import annotations

import os
import sys
from pathlib import Path


def add_mjlab_project(mjlab_root: str | Path) -> Path:
    """Make an existing ``unitree_rl_mjlab_lgdog`` checkout importable."""

    root = Path(mjlab_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    candidates = (
        root,
        root / "unitree_rl_mjlab_lgdog",
        root / "unitree_rl_mjlab_lgdog" / "unitree_rl_mjlab_lgdog",
    )
    project = next((p for p in candidates if (p / "src").exists()), None)
    if project is None:
        expected = " or ".join(str(p / "src") for p in candidates)
        raise FileNotFoundError(f"expected project source at {expected}")
    for p in (project, project.parent, root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return project


def make_lgdog_tracking_cfg(mjlab_root: str | Path, play: bool = False):
    """Build the existing ManagerBased tracking config with LgDog1 assets.

    This bridge imports mjlab lazily and raises an actionable error if the
    checkout or its Python environment is incomplete.
    """

    add_mjlab_project(mjlab_root)
    try:
        from mjlab.envs import ManagerBasedRlEnvCfg
        from mjlab.envs.mdp.actions import JointPositionActionCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from mjlab.sensor import ContactMatch, ContactSensorCfg
        from mjlab.tasks.tracking.mdp import MotionCommandCfg
        from src.assets.robots import LGDOG1_ACTION_SCALE, get_lgdog1_robot_cfg
        from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
    except ImportError as exc:
        raise RuntimeError("install the mjlab project's pinned environment before using this bridge") from exc

    cfg: ManagerBasedRlEnvCfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_lgdog1_robot_cfg()}
    cfg.scene.sensors = (ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found", "force"), reduce="none", num_slots=1, history_length=4,
    ),)
    action = cfg.actions["joint_pos"]
    if isinstance(action, JointPositionActionCfg):
        action.scale = LGDOG1_ACTION_SCALE
        action.actuator_names = (".*",)
    motion = cfg.commands["motion"]
    if not isinstance(motion, MotionCommandCfg):
        raise TypeError("tracking config did not create a MotionCommandCfg")
    motion.anchor_body_name = "trunk"
    motion.body_names = (
        "trunk", "FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf",
    )
    cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=("trunk",))
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^[FR][LR]_calf1_collision$"
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "FR_calf", "FL_calf", "RR_calf", "RL_calf"
    )
    cfg.viewer.body_name = "trunk"
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion.pose_range, motion.velocity_range = {}, {}
        motion.sampling_mode = "start"
    return cfg


def register_lgdog_tracking_task(mjlab_root: str | Path, task_id: str = "LgDog1-SuperMimic-Tracking") -> None:
    add_mjlab_project(mjlab_root)
    from mjlab.tasks.registry import register_mjlab_task
    from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner
    from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    register_mjlab_task(task_id=task_id, env_cfg=make_lgdog_tracking_cfg(mjlab_root),
                        play_env_cfg=make_lgdog_tracking_cfg(mjlab_root, play=True),
                        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
                        runner_cls=MotionTrackingOnPolicyRunner)

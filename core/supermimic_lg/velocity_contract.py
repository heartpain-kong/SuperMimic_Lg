"""Algorithm-independent training task description for controlled comparisons."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _depth_profile(args):
    """Depth configuration shared by both algorithms, or None when disabled."""
    if not getattr(args, "depth", False):
        return None
    from .depth import depth_spec
    return depth_spec(args.depth_size, args.depth_fov, args.depth_range, args.depth_feat,
                      noise=getattr(args, "depth_noise", 0.), dropout=getattr(args, "depth_dropout", 0.),
                      latency=getattr(args, "depth_latency", 0))


def _depth_feat(args):
    profile = _depth_profile(args)
    return 0 if profile is None else profile["feat"]


def shared_task_contract(xml,root,args):
    from .velocity_objective import objective_from, command_settings
    profile=None
    root=Path(root).resolve()
    candidates=(root,root/'unitree_rl_mjlab_lgdog',root/'unitree_rl_mjlab_lgdog/unitree_rl_mjlab_lgdog')
    constants=next((p/'src/assets/robots/lgdog1/lgdog1_constants.py' for p in candidates
                    if (p/'src/assets/robots/lgdog1/lgdog1_constants.py').is_file()),None)
    if args.terrain!='flat':
        path=next((p/'src/tasks/velocity/config/lgdog1/terrain_cfg.py' for p in candidates
                   if (p/'src/tasks/velocity/config/lgdog1/terrain_cfg.py').is_file()),None)
        if path is None:raise ValueError('missing LgDog terrain profile')
        profile=sha(path)
    contract=dict(version=1,xml_sha256=sha(xml),terrain=args.terrain,terrain_profile_sha256=profile,
        physics_backend=getattr(args,'backend','mjlab'),
        robot_constants_sha256=sha(constants) if constants else None,
        reward_and_failure_code_sha256=sha(Path(__file__).with_name('velocity_task.py')),
        physics_code_sha256=sha(Path(__file__).with_name('velocity_physics.py')),
        native_adapter_sha256=sha(Path(__file__).parent/'adapters/mjlab_velocity.py'),
        control_dt=.02,physics_dt=.005,actor_frame_dim=47+_depth_feat(args),history=args.history,
        critic_frame_dim=129+_depth_feat(args),depth=_depth_profile(args),
        command_limits=[args.command_vx,args.command_vy,args.command_yaw],command_resampling_seconds=5.,
        command_curriculum=dict(iterations=getattr(args,'command_curriculum_iterations',0),
            start=getattr(args,'command_curriculum_start',1.)),
        exploration=dict(noise=getattr(args,'explore_noise',0.),iterations=getattr(args,'explore_iterations',0),
            action_noise=getattr(args,'action_noise',0.)),
        command_settings=command_settings(args), objective=objective_from(args).as_dict(),
        objective_code_sha256=sha(Path(__file__).with_name('velocity_objective.py')),
        standing_fraction=command_settings(args)['standing_fraction'],episode_steps=args.episode_steps,reset_joint_noise=.003,
        reset_heading=0.,domain_randomization=False,observation_noise=False,seed=args.seed,
        action_scale=.25,clip_actions=1.,reward_scaled_by_dt=False)
    return contract,hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()

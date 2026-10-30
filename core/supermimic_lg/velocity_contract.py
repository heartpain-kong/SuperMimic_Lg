"""Algorithm-independent training task description for controlled comparisons."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shared_task_contract(xml,root,args):
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
        control_dt=.02,physics_dt=.005,actor_frame_dim=47,history=args.history,critic_frame_dim=129,
        command_limits=[args.command_vx,args.command_vy,args.command_yaw],command_resampling_seconds=5.,
        standing_fraction=.1,episode_steps=args.episode_steps,reset_joint_noise=.003,
        reset_heading=0.,domain_randomization=False,observation_noise=False,seed=args.seed,
        action_scale=.25,clip_actions=1.,reward_scaled_by_dt=False)
    return contract,hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()

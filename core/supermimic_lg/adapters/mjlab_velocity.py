"""Run the existing LgDog HIM/PPO learner on the shared velocity task."""
from copy import deepcopy
import numpy as np
import torch

from supermimic_lg.simulator import joint_addresses, resolve_robot_xml
from supermimic_lg.velocity_physics import scene_config, velocity_sensors, gpu_observation
from supermimic_lg.velocity_task import (FootKinematics, reward, failure, phase_at,
    state_features, velocity_error)
from supermimic_lg.state import heading_quat, quat_apply


def snapshot(env):
    # Native manager env normally postpones forward until after reward/reset.
    # Refresh once here to use the same post-step geometry as the MBRL backend.
    cache=getattr(env,"_shared_velocity_snapshot",None)
    if cache is None or cache[0] != env.common_step_counter:
        env.sim.forward();env.scene.update(0.);env.sim.sense()
        qa,va=joint_addresses(env.sim.mj_model)
        values=gpu_observation(env.scene,env.sim,qa,va,env.num_envs)
        env._shared_velocity_snapshot=(env.common_step_counter,values)
    return env._shared_velocity_snapshot[1]


def physical_failure(env):
    return failure(*snapshot(env))


def privileged_observation(env):
    state,context=snapshot(env)
    command=env.command_manager.get_command("twist")
    phase=phase_at(env.episode_length_buf)
    phase *= (command.norm(dim=-1)>=.1)[:,None]
    return torch.cat((state_features(state,context),command,phase),-1)


def xy_error(env):
    return velocity_error(snapshot(env)[0],env.command_manager.get_command("twist"))[0]


def shared_reward(env, xml):
    state,context=snapshot(env)
    if not hasattr(env,"_shared_velocity_feet"):
        env._shared_velocity_feet=FootKinematics(xml,env.device)
    command=env.command_manager.get_command("twist")
    phase=phase_at((env.episode_length_buf-1).clamp_min(0))
    phase *= (command.norm(dim=-1)>=.1)[:,None]
    stats=getattr(env,"_shared_velocity_stats",None)
    if stats is not None and stats['last_step'] != env.common_step_counter:
        desired=quat_apply(heading_quat(state[:,3:7]),torch.cat((command[:,:2],command[:,:1]*0),-1))[:,:2]
        speed=desired.norm(dim=-1)
        stats['progress'] += ((state[:,:2]-stats['previous_xy'])*desired/speed.clamp_min(1e-6)[:,None]).sum(-1)
        stats['target'] += speed*.02
        stats['error'] += velocity_error(state,command)[0]
        stats['steps'] += 1
        stats['previous_xy'][:] = state[:,:2]
        stats['last_step'] = env.common_step_counter
    return reward(state,env.action_manager.action,env.action_manager.prev_action,
                  command,phase,context,env._shared_velocity_feet)


def reset_shared(env, env_ids):
    if env_ids is None: env_ids=torch.arange(env.num_envs,device=env.device)
    qa,va=joint_addresses(env.sim.mj_model)
    q=env.sim.data.qpos[env_ids].clone();v=env.sim.data.qvel[env_ids].clone()
    q[:,:3]=env.scene.env_origins[env_ids];q[:,2]+=.31
    q[:,3:7]=q.new_tensor([1.,0.,0.,0.])
    from supermimic_lg.config import lgdog1_spec
    default=q.new_tensor(lgdog1_spec().default_joint_pos)
    if not hasattr(env,'_shared_velocity_rng'):
        env._shared_velocity_rng=np.random.default_rng(env.cfg.seed)
    q[:,qa]=default+torch.tensor(env._shared_velocity_rng.uniform(-.003,.003,(len(env_ids),12)),
                                device=env.device,dtype=q.dtype)
    v[:]=0.
    env.sim.data.qpos[env_ids]=q;env.sim.data.qvel[env_ids]=v
    robot=env.scene['robot']
    jids=torch.tensor([robot.joint_names.index(name) for name in lgdog1_spec().joint_names],device=env.device)
    robot.set_joint_position_target(q[:,qa],joint_ids=jids,env_ids=env_ids[:,None])
    if not hasattr(env,'_shared_velocity_stats'):
        zero=torch.zeros(env.num_envs,device=env.device)
        env._shared_velocity_stats=dict(progress=zero.clone(),target=zero.clone(),error=zero.clone(),
            steps=zero.clone(),previous_xy=torch.zeros(env.num_envs,2,device=env.device),last_step=-1)
    stats=env._shared_velocity_stats
    for key in ('progress','target','error','steps'):stats[key][env_ids]=0.
    stats['previous_xy'][env_ids]=q[:,:2]
    env._shared_velocity_snapshot=None


def curriculum(env,env_ids):
    stats=getattr(env,'_shared_velocity_stats',None)
    terrain=env.scene['terrain']
    if stats is None or getattr(terrain,'terrain_origins',None) is None:return 0.
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    eligible=(stats['steps'][env_ids]>=50)&(stats['target'][env_ids]>=2.)
    failed=env.reset_terminated[env_ids]
    progress,target=stats['progress'][env_ids],stats['target'][env_ids]
    up=(~failed & eligible & (progress>=2.) & (progress>=.7*target)
        & (stats['error'][env_ids]/stats['steps'][env_ids].clamp_min(1)<.8))
    down=failed | (eligible & (progress<.4*target))
    terrain.update_env_origins(env_ids,up,down)
    return terrain.terrain_levels.float().mean()


def make_him_task(root,terrain='flat',n=64,seed=42,history=3,width=128,
                  command_vx=.6,command_vy=.3,command_yaw=.6,episode_steps=500,collect_steps=16):
    from .mjlab_tracking import add_mjlab_project
    add_mjlab_project(root)
    from src.tasks.velocity.config.lgdog1.him_env_cfg import lgdog1_him_standard_env_cfg,lgdog1_him_runner_cfg
    from mjlab.managers import RewardTermCfg,TerminationTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.envs import mdp
    from mjlab.sim import SimulationCfg,MujocoCfg
    cfg=lgdog1_him_standard_env_cfg()
    cfg.scene=scene_config(root,terrain,n,seed,'cuda:0',velocity_sensors())
    cfg.sim=SimulationCfg(mujoco=MujocoCfg(timestep=.005,ccd_iterations=500),
        nconmax=35 if terrain!='flat' else None,contact_sensor_maxmatch=500)
    cfg.seed,cfg.decimation,cfg.episode_length_s=seed,4,episode_steps*.02
    cfg.scale_rewards_by_dt=False
    cfg.events={'reset_shared':EventTermCfg(func=reset_shared,mode='reset')}
    cfg.terminations={'time_out':TerminationTermCfg(func=mdp.time_out,time_out=True),
                      'physical_failure':TerminationTermCfg(func=physical_failure)}
    cfg.rewards={'velocity_task':RewardTermCfg(func=shared_reward,weight=1.,params={'xml':str(resolve_robot_xml(root))})}
    cfg.curriculum={'terrain':CurriculumTermCfg(func=curriculum)} if terrain!='flat' else {}
    cfg.metrics={'velocity_error_xy_mps':MetricsTermCfg(func=xy_error)}
    command=cfg.commands['twist']
    command.resampling_time_range=(5.,5.);command.rel_standing_envs=.1;command.heading_command=False
    command.init_velocity_prob=0.
    command.ranges.lin_vel_x=(-command_vx,command_vx)
    command.ranges.lin_vel_y=(-command_vy,command_vy)
    command.ranges.ang_vel_z=(-command_yaw,command_yaw)
    actor=cfg.observations['actor'];actor.history_length=history;actor.enable_corruption=False
    cfg.observations['him_target']=ObservationGroupCfg(terms=deepcopy(actor.terms),concatenate_terms=True,
        enable_corruption=False,history_length=1,flatten_history_dim=True)
    cfg.observations['critic']=ObservationGroupCfg(terms={'privileged':ObservationTermCfg(func=privileged_observation)},
        concatenate_terms=True,enable_corruption=False,history_length=1)
    agent=lgdog1_him_runner_cfg('lgdog_velocity_shared_him')
    agent.actor.history_length=history
    agent.actor.hidden_dims=(width,width);agent.critic.hidden_dims=(width,width)
    agent.num_steps_per_env=collect_steps;agent.seed=seed;agent.clip_actions=1.
    agent.logger='tensorboard';agent.upload_model=False
    return cfg,agent

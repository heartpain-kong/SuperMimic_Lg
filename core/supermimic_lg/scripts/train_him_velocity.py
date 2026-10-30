"""Existing HIM/PPO learner, shared velocity rewards/physics/terrain protocol."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from supermimic_lg.velocity_physics import TERRAINS
from supermimic_lg.benchmark import hardware,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mjlab-root',type=Path,required=True)
    p.add_argument('--terrain',choices=TERRAINS,default='flat')
    p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--iterations',type=int,default=20000)
    p.add_argument('--history',type=int,default=3)
    p.add_argument('--network-size',type=int,default=128)
    p.add_argument('--collect-steps',type=int,default=16)
    p.add_argument('--episode-steps',type=int,default=500)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--command-vx',type=float,default=.6)
    p.add_argument('--command-vy',type=float,default=.3)
    p.add_argument('--command-yaw',type=float,default=.6)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--torch-threads',type=int,default=4)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--check-config',action='store_true',help='construct/validate task configuration without launching Warp')
    a=p.parse_args()
    if any(getattr(a,k)<1 for k in ('num_envs','iterations','history','network_size','collect_steps','episode_steps','torch_threads')):
        p.error('counts must be positive')
    if any(not 0<=getattr(a,k)<=3 for k in ('command_vx','command_vy','command_yaw')):p.error('invalid command limits')
    if (a.output_dir/'config.json').exists():raise FileExistsError('choose a fresh output directory')
    import torch
    torch.set_num_threads(a.torch_threads)
    from supermimic_lg.adapters.mjlab_velocity import make_him_task
    cfg,agent=make_him_task(a.mjlab_root,a.terrain,a.num_envs,a.seed,a.history,a.network_size,
                           a.command_vx,a.command_vy,a.command_yaw,a.episode_steps,a.collect_steps)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    config=dict(task='lgdog_velocity_v1',algorithm='existing_HIM_PPO',args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        reward_source_sha256=digest(Path(__file__).parents[1]/'velocity_task.py'),
        actor_frame_dim=47,critic_dim=129,dt=.02,physics_dt=.005,domain_randomization=False,
        observation_noise=False,reward_scale_by_dt=False)
    config['learning_precision']='tf32'
    from supermimic_lg.velocity_contract import shared_task_contract
    from supermimic_lg.simulator import resolve_robot_xml
    contract,contract_sha=shared_task_contract(resolve_robot_xml(a.mjlab_root),a.mjlab_root,a)
    config.update(shared_task_contract=contract,shared_task_contract_sha256=contract_sha)
    (a.output_dir/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    if a.check_config:
        print(json.dumps(config,indent=2));return
    import torch
    if not a.device.startswith('cuda:') or not torch.cuda.is_available():p.error('HIM/PPO shared-task training requires CUDA')
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    from src.tasks.velocity.rl.runner import PPOHIMRunner
    configure_torch_backends()
    agent.max_iterations=a.iterations
    env=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg,device=a.device),clip_actions=1.)
    try:
        dump_yaml(a.output_dir/'params/env.yaml',asdict(cfg));dump_yaml(a.output_dir/'params/agent.yaml',asdict(agent))
        runner=PPOHIMRunner(env,asdict(agent),str(a.output_dir),a.device)
        runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
    finally:
        env.close()


if __name__=='__main__':main()

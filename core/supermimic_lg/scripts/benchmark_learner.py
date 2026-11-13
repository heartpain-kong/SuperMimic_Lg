"""Time identical learner updates on a fixed real-robot state batch.

This isolates implementation throughput, not collection speed or convergence.
Run the same command on each revision, with the same device and precision.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-xml', type=Path)
    p.add_argument('--mjlab-root', type=Path, default=Path('../unitree_rl_mjlab_lgdog'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    p.add_argument('--algorithm', choices=('xmimic', 'supermimic'), default='xmimic')
    p.add_argument('--rollout-mode', choices=('batched', 'reference', 'compiled', 'auto'), default='batched')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--horizon', type=int, default=32)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--updates', type=int, default=3)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--profile', action='store_true', help='record operator counts in one additional update')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if min(args.batch_size, args.horizon, args.updates, args.threads) < 1 or args.warmup < 0:
        p.error('positive sizes/counts required; warmup >=0')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    from supermimic_lg.benchmark import hardware
    from supermimic_lg.simulator import resolve_robot_xml
    from supermimic_lg.config import lgdog1_spec
    from supermimic_lg.tasks.velocity.networks import VelocityWorldModel, VelocityPolicy, VelocityCritic
    from supermimic_lg.tasks.velocity.mdp import FootKinematics, frame, phase_at
    from supermimic_lg.tasks.velocity.task import VelocityTask
    from supermimic_lg.learning.velocity import VelocityLearner, VelocityLearningConfig
    from supermimic_lg.xmimic_lg.learner import XMimicLearner, XMimicLearningConfig
    cls, config = ((XMimicLearner, XMimicLearningConfig) if args.algorithm == 'xmimic'
                   else (VelocityLearner, VelocityLearningConfig))
    wm = VelocityWorldModel((1024, 1024, 1024)).to(args.device)
    actor = VelocityPolicy(4, (512, 256, 128)).to(args.device)
    critic = VelocityCritic((256, 128)).to(args.device)
    feet = FootKinematics(resolve_robot_xml(args.mjlab_root, args.robot_xml), args.device)
    cfg = config(policy_batch_size=args.batch_size, horizon=args.horizon,
                 critic_epochs=5, critic_minibatches=4)
    learner = cls(wm, actor, critic, VelocityTask(feet), cfg)
    learner.rollout_mode = args.rollout_mode
    state = torch.zeros(args.batch_size, 42, device=args.device)
    state[:, 2], state[:, 3] = .3, 1.
    state[:, 7:19] = state.new_tensor(lgdog1_spec().default_joint_pos)
    state[:, 37:41] = 1.
    context = state.new_zeros(args.batch_size, 85)
    previous = state.new_zeros(args.batch_size, 12)
    command = state.new_tensor([.4, 0., 0.]).expand(args.batch_size, -1)
    phase = phase_at(state.new_zeros(args.batch_size))
    history = frame(state, previous, command, phase)[:, None].repeat(1, 4, 1)
    rows = {k:v[None] for k,v in dict(state=state, context=context, previous=previous,
                                    command=command, phase=phase, history=history).items()}
    with torch.no_grad():
        wm.update_norms(state, previous, context, state)
        actor.obs_norm.update(history)

    def sync():
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize(args.device)

    def update():
        amp = torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16) if args.precision == 'bf16' else nullcontext()
        with amp:
            return learner.update_policy(rows)

    for _ in range(args.warmup):
        update()
    sync()
    stage_seconds = []
    if hasattr(learner, 'timings'):
        learner.timings.reset()
    times = []
    for _ in range(args.updates):
        start = time.perf_counter()
        metrics = update()
        sync()
        times.append(time.perf_counter()-start)
        if hasattr(learner, 'timings'):
            stage_seconds.append(learner.timings.metrics(synchronize=False))
    report = dict(protocol='fixed LgDog standing-state batch, full actor backward plus 20 critic steps per update',
        hardware=hardware(), device=args.device, precision=args.precision, algorithm=args.algorithm, rollout_mode=args.rollout_mode,
        selected_rollout_mode=learner.rollout_mode,runtime_tuning=learner.runtime_tuning_report,
        config=asdict(cfg), warmup=args.warmup, timed_updates=args.updates, threads=args.threads,
        seconds=times, median_s=statistics.median(times), stage_seconds=stage_seconds, updates=dict(learner.counts),
        metrics={k:float(v) for k,v in metrics.items()},
        limitations='Fixed-batch implementation benchmark; does not measure locomotion quality or simulation throughput')
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if str(args.device).startswith('cuda'):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as prof:
            update()
            sync()
        report['aten_calls'] = sum(e.count for e in prof.key_averages() if e.key.startswith('aten::'))
        report['cpu_profile'] = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=15)
        if str(args.device).startswith('cuda'):
            events = prof.key_averages()
            sort = 'self_device_time_total' if hasattr(events[0], 'self_device_time_total') else 'self_cuda_time_total'
            report['cuda_profile'] = events.table(sort_by=sort, row_limit=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(median_s=report['median_s'], seconds=times, updates=report['updates'],
                          aten_calls=report.get('aten_calls'), output=str(args.output))))


if __name__ == '__main__':
    main()

"""Command-line configuration and dispatch for tracking and velocity training."""
import argparse
import math
from pathlib import Path
import signal
import sys
import torch


class TrainingArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        from .xmimic_lg.profile import resolve_profile
        argv = list(sys.argv[1:] if args is None else args)
        return resolve_profile(super().parse_args(argv, namespace), argv)


def parser():
    p = TrainingArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=("supermimic", "xmimic"), default="supermimic",
                   help="xmimic is the PyTorch LgDog velocity adaptation; native JAX has a separate launcher")
    p.add_argument("--warmup-steps", type=int, default=0, help="XMimic: initial vectorized physics steps before fitting")
    for flag in ("policy-ema-decay", "wm-mirror-fraction", "policy-lr-mult"):
        p.add_argument("--"+flag, type=float)
    for flag in ("lr-decay-iterations", "policy-lr-decay-start", "policy-lr-decay-length"):
        p.add_argument("--"+flag, type=int)
    p.add_argument("--mjlab-root", type=Path, default=Path("../unitree_rl_mjlab_lgdog"))
    p.add_argument("--robot-xml", type=Path)
    p.add_argument("--task", choices=("tracking", "velocity"), default="tracking")
    p.add_argument('--rollout-mode', choices=('batched', 'reference', 'compiled'), default='batched',
                   help='velocity: batched geometry (default), sequential reference, or experimental compiled recurrent step')
    p.add_argument("--motion-file", type=Path)
    p.add_argument("--backend", choices=("mjlab", "mujoco"), default="mjlab")
    from supermimic_lg.velocity_physics import TERRAINS
    p.add_argument("--terrain", choices=TERRAINS, default="flat",
                   help="flat plane or LgDog mjlab stairs/slopes/rough/wave terrain (mjlab backend only)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--wm-batch-size", type=int, help="WM window batch; defaults to --batch-size")
    p.add_argument("--policy-batch-size", type=int, help="imagined trajectory batch; defaults to --batch-size")
    p.add_argument("--update-to-data-ratio", type=float,
                   help="optional iterations per vectorized physics step, in (0,1]; overrides fixed collect-steps")
    for flag in ("wm-lr", "actor-lr", "critic-lr", "gamma", "gae-lambda", "actor-beta2",
                 "actor-grad-value", "reward-limit", "value-target-limit"):
        p.add_argument("--"+flag, type=float, help="optional learner override; resolved settings are saved")
    p.add_argument("--critic-epochs", type=int)
    p.add_argument("--critic-minibatches", type=int)
    actor_scale = p.add_mutually_exclusive_group()
    actor_scale.add_argument("--normalize-actor-loss", dest="normalize_actor_loss", action="store_true", default=None)
    actor_scale.add_argument("--no-normalize-actor-loss", dest="normalize_actor_loss", action="store_false")
    for name, default in (("num-envs", 64), ("iterations", 10000), ("collect-steps", 16), ("wm-updates", 10),
                          ("policy-updates", 2), ("warmup-iterations", 5), ("batch-size", 128), ("wm-horizon", 4),
                          ("horizon", 8), ("history", 3), ("goal-horizon", 10), ("replay-capacity", 256),
                          ("episode-steps", 500), ("save-interval", 10), ("torch-threads", 4)):
        p.add_argument("--" + name, type=int, default=default)
    from supermimic_lg.config import parse_hidden
    p.add_argument("--wm-hidden", type=lambda text: parse_hidden(text, "wm-hidden"),
                   default=(1024, 1024, 1024), help="world-model hidden widths, comma separated")
    p.add_argument("--actor-hidden", type=lambda text: parse_hidden(text, "actor-hidden"),
                   default=(512, 256, 128), help="policy hidden widths, comma separated")
    p.add_argument("--critic-hidden", type=lambda text: parse_hidden(text, "critic-hidden"),
                   default=(512, 256, 128), help="critic hidden widths, comma separated")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action-noise", type=float, default=.1)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                   help="velocity: bf16 autocast for neural networks; state integration stays float32")
    p.add_argument("--wm-loss-scale", choices=("fixed", "increment"), default="fixed",
                   help="velocity: increment is experimental; improved velocity prediction has not established policy gains")
    p.add_argument("--action-bound-weight", type=float, default=1.,
                   help="velocity: penalty on raw actor outputs outside executed [-1,1] limits")
    p.add_argument("--max-training-seconds", type=float, default=0.,
                   help="velocity: stop and save after this invocation's wall budget (0 disables; checked per iteration)")
    p.add_argument("--explore-noise", type=float, default=1.0,
                   help="action-noise scale at the start of training; decays to --action-noise")
    p.add_argument("--command-curriculum-iterations", type=int, default=500,
                   help="iterations to ramp command magnitude from --command-curriculum-start to 1.0 (0 disables)")
    p.add_argument("--command-curriculum-start", type=float, default=0.25,
                   help="command scale at iteration 0 of the curriculum")
    p.add_argument("--explore-iterations", type=int, default=100,
                   help="iterations over which exploration noise decays (0 disables the schedule)")
    p.add_argument("--depth", action="store_true",
                   help="velocity only: add an egocentric pinhole depth camera to the actor and world model")
    p.add_argument("--depth-size", default="18x24", help="depth image height x width in rays")
    p.add_argument("--depth-fov", type=float, default=60., help="vertical field of view, degrees")
    p.add_argument("--depth-range", type=float, default=3., help="maximum sensed distance, metres")
    p.add_argument("--depth-feat", type=int, default=32, help="depth encoder output dimension")
    p.add_argument("--depth-noise", type=float, default=0.,
                   help="quadratic range noise coefficient, sigma = noise * z^2 (metres)")
    p.add_argument("--depth-dropout", type=float, default=0., help="fraction of invalid depth pixels")
    p.add_argument("--depth-latency", type=int, default=0, help="depth sensor delay in control steps")
    p.add_argument("--output-dir", type=Path, default=Path("logs/supermimic_lgdog"))
    p.add_argument("--resume", type=Path)
    p.add_argument("--init-policy", type=Path, help="velocity only: transfer actor weights, restart world model/critic/counters")
    p.add_argument("--command-vx", type=float, default=.6, help="absolute planar forward command limit, m/s")
    p.add_argument("--command-vy", type=float, default=.3)
    p.add_argument("--command-yaw", type=float, default=.6)
    from supermimic_lg.velocity_objective import add_objective_arguments
    add_objective_arguments(p)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    positive = ("num_envs", "iterations", "collect_steps", "wm_updates", "policy_updates", "warmup_iterations", "batch_size",
                "wm_horizon", "horizon", "history", "goal_horizon", "replay_capacity", "episode_steps", "save_interval", "torch_threads")
    if any(getattr(args, x) <= 0 for x in positive) or args.replay_capacity < args.wm_horizon or not 0 <= args.action_noise <= 1:
        p.error("sizes/counts must be positive, replay capacity >= wm horizon, action noise in [0,1]")
    from supermimic_lg.learning.velocity import VelocityLearningConfig
    from supermimic_lg.learning.schedule import CollectionSchedule
    try:
        if args.algorithm == "xmimic":
            from .xmimic_lg.learner import XMimicLearningConfig
            XMimicLearningConfig.from_args(args)
            if args.task != "velocity" or args.depth:
                p.error("XMimic-Lg currently requires --task velocity without --depth; native JAX G1 uses xmimic_native")
            if args.warmup_steps < 0:
                p.error("warmup-steps must be non-negative")
        else:
            VelocityLearningConfig.from_args(args)
            special = ("policy_ema_decay", "wm_mirror_fraction", "policy_lr_mult", "lr_decay_iterations",
                       "policy_lr_decay_start", "policy_lr_decay_length")
            if args.warmup_steps or any(getattr(args, key) is not None for key in special):
                p.error("XMimic-specific options require --algorithm xmimic")
        CollectionSchedule(args.collect_steps, args.update_to_data_ratio)
    except ValueError as exc:
        p.error(str(exc))
    if args.command_curriculum_iterations < 0 or not 0 < args.command_curriculum_start <= 1:
        p.error("command curriculum needs iterations >= 0 and 0 < start <= 1")
    if args.explore_iterations < 0 or args.explore_noise < 0:
        p.error("exploration iterations and noise must be non-negative")
    if not math.isfinite(args.action_bound_weight) or args.action_bound_weight < 0:
        p.error("action-bound-weight must be finite and non-negative")
    if not math.isfinite(args.max_training_seconds) or args.max_training_seconds < 0:
        p.error("max-training-seconds must be finite and non-negative")
    if args.task != "velocity" and args.max_training_seconds:
        p.error("max-training-seconds is currently available for velocity training")
    if args.precision == "bf16" and (args.task != "velocity" or not args.device.startswith("cuda")):
        p.error("bf16 training is available for velocity with a CUDA device")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        p.error("this CUDA device does not support bf16")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA is unavailable; use --backend mujoco --device cpu for real CPU physics")
    if args.task == "tracking" and args.motion_file is None:
        p.error("--motion-file is required for --task tracking; velocity needs no motion file")
    if args.init_policy and (args.task != "velocity" or args.resume):
        p.error("--init-policy is velocity-only and mutually exclusive with --resume")
    if args.depth:
        if args.task != "velocity":
            p.error("--depth is available in velocity mode; tracking uses reference motions")
        from supermimic_lg.depth import depth_spec
        try:
            depth_spec(args.depth_size, args.depth_fov, args.depth_range, args.depth_feat,
                       noise=args.depth_noise, dropout=args.depth_dropout, latency=args.depth_latency)
        except (ValueError, TypeError) as exc:
            p.error(str(exc))
    if args.task == "tracking" and args.terrain not in ("flat", "rough"):
        p.error("tracking supports flat/rough; named terrain families are available in velocity mode")
    if args.task == "tracking" and args.terrain == "rough" and args.backend != "mjlab":
        p.error("--terrain rough requires --backend mjlab")
    from supermimic_lg.velocity_objective import objective_from, command_settings
    try:
        objective_from(args)
        command_settings(args)
    except ValueError as exc:
        p.error(str(exc))
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    if args.task == "velocity":
        from .runners.velocity import VelocityTrainer
        VelocityTrainer(args).run()
    else:
        from .runners.tracking import Trainer
        Trainer(args).run()


if __name__ == "__main__":
    main()

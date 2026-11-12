"""Noise-free real-physics checkpoint evaluation, including a zero-action baseline."""
import argparse
import json
import math
from pathlib import Path

import torch

from supermimic_lg.benchmark import CheckpointActor, digest, evaluate_actor, hardware
from supermimic_lg.simulator import resolve_robot_xml
from supermimic_lg.velocity_physics import TERRAINS, VelocitySimulator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--policy", choices=("auto", "raw", "ema"), default="auto")
    p.add_argument("--mjlab-root", type=Path, default=Path("../unitree_rl_mjlab_lgdog"))
    p.add_argument("--robot-xml", type=Path)
    p.add_argument("--terrain", choices=[t for t in TERRAINS if t != "rough"], default="flat")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--seconds", type=float, default=10.)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    p.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "YAW"),
                   help="fixed command; omitted means the shared benchmark command schedule")
    p.add_argument("--backend", choices=("mujoco", "mjlab"), default="mujoco")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 5 or (args.command and
            not all(math.isfinite(x) for x in args.command)):
        p.error("use >=5 finite seconds and finite commands")
    torch.set_num_threads(1)
    xml = resolve_robot_xml(args.mjlab_root, args.robot_xml)
    actor = CheckpointActor(args.checkpoint, args.device, xml, args.policy)

    class ZeroActor:
        history = actor.history

        def encode(self, sim):
            return actor.encode(sim)

        def __call__(self, obs):
            return obs.new_zeros(len(obs), 12)

    sim = VelocitySimulator(xml, args.mjlab_root, 1, args.backend, args.device,
                            args.terrain, seed=0, level=args.level, depth=actor.depth)
    report = dict(algorithm=actor.algorithm, policy_source=actor.policy_source,
        checkpoint_sha256=digest(args.checkpoint), xml_sha256=digest(xml),
        iteration=actor.iteration, training_seconds=actor.training_seconds, hardware=hardware(),
        protocol=dict(terrain=args.terrain, level=args.level, seeds=args.seeds, seconds=args.seconds,
            fixed_command=args.command, backend=args.backend, action_noise=0.,
            depth_sensor="ideal; no noise/dropout/latency" if actor.depth else None), results={})
    try:
        for name, evaluated in (("policy", actor), ("zero_action", ZeroActor())):
            episodes = evaluate_actor(evaluated, sim, args.seeds, args.seconds, args.command)
            metrics = ("velocity_error_xy_mps", "yaw_error_rps", "command_aligned_progress_m",
                       "survival_seconds", "action_saturation", "raw_action_abs")
            report["results"][name] = {key: sum(row[key] for row in episodes)/len(episodes) for key in metrics}
            report["results"][name].update(episodes=episodes,
                failure_rate=sum(row["failed"] for row in episodes)/len(episodes))
    finally:
        sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: {m: v for m, v in row.items() if m != "episodes"}
                      for k, row in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()

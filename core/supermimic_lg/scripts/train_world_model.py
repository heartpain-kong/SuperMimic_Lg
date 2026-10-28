from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from supermimic_lg.config import WorldModelConfig, lgdog1_spec
from supermimic_lg.data.replay import ReplayStore
from supermimic_lg.training import WorldModelTrainer
from supermimic_lg.world_model import WorldModel


def main() -> None:
    p = argparse.ArgumentParser(description="Fit the SuperMimic world model from recorded transitions")
    p.add_argument("replay", help="NPZ with states/actions/contexts/[done]")
    p.add_argument("--robot", choices=("lgdog1",), default="lgdog1")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--context-dim", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, default=Path("logs/world_model/model.pt"))
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=100)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.horizon, args.log_interval, args.save_interval) <= 0:
        p.error("steps/batch-size/horizon/log-interval/save-interval must be positive")
    robot = lgdog1_spec()
    store = ReplayStore.from_npz(args.replay, args.device)
    model = WorldModel(robot, args.context_dim, WorldModelConfig(), args.device)
    trainer = WorldModelTrainer(model, args.lr)
    last = {}
    completed = 0
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        torch.save({"model": model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                    "step": completed, "metrics": last, "robot": asdict(robot),
                    "world_model_config": asdict(WorldModelConfig()),
                    "context_dim": model.context_dim}, tmp)
        tmp.replace(args.output)
        print(json.dumps({"checkpoint": str(args.output.resolve()), "step": completed}), flush=True)
    try:
        for step in range(1, args.steps + 1):
            last = trainer.update(store.sample(args.batch_size, args.horizon))
            completed = step
            if step == 1 or step % args.log_interval == 0 or step == args.steps:
                print(json.dumps({"step": step, "steps": args.steps, **last}), flush=True)
            if step % args.save_interval == 0:
                save()
    except KeyboardInterrupt:
        print("Stop requested; saving the latest world model.", flush=True)
    finally:
        if completed:
            save()


if __name__ == "__main__":
    main()

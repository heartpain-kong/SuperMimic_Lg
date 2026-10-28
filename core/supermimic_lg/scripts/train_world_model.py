from __future__ import annotations

import argparse
import json

from supermimic_lg.adapters.lgdog1 import LGDOG1
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
    args = p.parse_args()
    robot = lgdog1_spec()
    store = ReplayStore.from_npz(args.replay, args.device)
    model = WorldModel(robot, args.context_dim, WorldModelConfig(), args.device)
    trainer = WorldModelTrainer(model, args.lr)
    last = {}
    for _ in range(args.steps):
        last = trainer.update(store.sample(args.batch_size, args.horizon))
    print(json.dumps(last, indent=2))


if __name__ == "__main__":
    main()

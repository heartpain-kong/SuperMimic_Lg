from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from supermimic_lg.config import lgdog1_spec, unitree_go2_spec
from supermimic_lg.synthetic import make_replay_window
from supermimic_lg.training import WorldModelTrainer
from supermimic_lg.world_model import WorldModel, WorldModelConfig


def main() -> None:
    p = argparse.ArgumentParser(description="CPU smoke test for the generic SuperMimic world model")
    p.add_argument("--robot", choices=("lgdog1", "go2"), default="lgdog1")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    robot = lgdog1_spec() if args.robot == "lgdog1" else unitree_go2_spec()
    model = WorldModel(robot, context_dim=None, cfg=WorldModelConfig(hidden=(64, 64), term_hidden=(16, 16)), device=args.device)
    window = make_replay_window(robot, args.batch_size, args.horizon, device=args.device)
    before = model.rollout_error(window.states, window.actions, window.contexts)
    trainer = WorldModelTrainer(model, lr=1e-3)
    metrics = {}
    for _ in range(args.steps):
        metrics = trainer.update(window)
    after = model.rollout_error(window.states, window.actions, window.contexts)
    # Check that the imagined dynamics graph reaches the action tensor.
    action = window.actions.clone().requires_grad_(True)
    state = window.states[0]
    for t in range(args.horizon):
        state = model.step(state, action[t], window.contexts)
    state.square().mean().backward()
    result = {
        "robot": robot.name, "state_dim": model.spec.state_dim, "action_dim": model.action_dim,
        "before_error": before.detach().cpu().tolist(), "after_error": after.detach().cpu().tolist(),
        "action_grad_l1": float(action.grad.abs().sum().detach().cpu()), "metrics": metrics,
    }
    print(json.dumps(result, indent=2))
    if not torch.isfinite(action.grad).all() or float(action.grad.abs().sum()) == 0.0:
        raise SystemExit("BPTT action gradient check failed")


if __name__ == "__main__":
    main()

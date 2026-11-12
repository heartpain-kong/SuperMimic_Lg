"""Paired normalization ablation on real MuJoCo data and separate validation worlds.

The legacy arm reproduces only the former update_norms bug, not an old trainer.
This checks prediction quality; it cannot establish locomotion convergence.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from supermimic_lg.learning.losses import contact_bce

from supermimic_lg.simulator import resolve_robot_xml
from supermimic_lg.velocity_physics import VelocitySimulator
from supermimic_lg.velocity_task import VelocityReplay, VelocityWorldModel, failure


def collect(xml, steps, seed):
    torch.manual_seed(seed)
    sim = VelocitySimulator(xml, Path("."), 8, "mujoco", "cpu", seed=seed)
    train, valid = [VelocityReplay(steps, 4, 1, "cpu") for _ in range(2)]
    try:
        state, context = sim.observe()
        for i in range(steps):
            action = torch.randn(8, 12).clamp(-1., 1.)
            sim.step(action)
            nxt, next_context = sim.observe()
            reset = failure(nxt, next_context) | (torch.full((8,), i % 64 == 63))
            for replay, ids in ((train, slice(0, 4)), (valid, slice(4, 8))):
                replay.add(state=state[ids], next=nxt[ids], context=context[ids],
                           action=action[ids], reset=reset[ids].float())
            sim.reset(reset.nonzero().flatten())
            state, context = sim.observe()
    finally:
        sim.close()
    return train, valid


@torch.no_grad()
def evaluate(wm, replay):
    metrics = {}
    for horizon in (1, 8, 16):
        torch.manual_seed(9000 + horizon)
        rows = replay.sample(256, horizon, fields=["state", "next", "context", "action"])
        state = rows["state"][0]
        errors, velocity = [], []
        for t in range(horizon):
            state = wm.step(state, rows["action"][t], rows["context"][0])
            errors.append(wm.spec.normalized_sq_error(state[:, :37], rows["next"][t, :, :37]))
            velocity.append((state[:, 19:22]-rows["next"][t, :, 19:22]).square().mean(-1))
        mask = rows["valid"]
        metrics[f"state_error_{horizon}"] = float((torch.stack(errors)*mask).sum()/mask.sum().clamp_min(1))
        metrics[f"velocity_rmse_{horizon}"] = float(((torch.stack(velocity)*mask).sum()/mask.sum().clamp_min(1)).sqrt())
    return metrics


def fit(replay, validation, updates, seed, legacy, adaptive=False):
    torch.manual_seed(seed)
    wm = VelocityWorldModel((64, 64))
    from supermimic_lg.norm import RunningNorm
    error_norm = RunningNorm(36, eps=1e-4)
    fixed_scale = torch.tensor([.05]*3 + [.1]*3 + [.5]*3 + [1.]*3 + [.1]*12 + [2.]*12)
    floor = torch.tensor([.003]*3 + [.005]*3 + [.02]*3 + [.05]*3 + [.003]*12 + [.05]*12)
    optimizer = torch.optim.AdamW(wm.parameters(), lr=4e-4)
    started = time.monotonic()
    for _ in range(updates):
        rows = replay.sample(128, 4, fields=["state", "next", "context", "action"])
        state, context = rows["state"][0], rows["context"][0]
        if legacy:
            with torch.no_grad():
                wm.in_norm.update(wm.condition(state, rows["action"][0], context))
                wm.out_norm.update(wm.spec.delta_between(state, rows["next"][0]))
        else:
            wm.update_norms(state, rows["action"][0], context, rows["next"][0])
        if adaptive:
            with torch.no_grad():
                error_norm.update(wm.spec.error_vector(rows["next"][0, :, :37], state[:, :37])*fixed_scale)
        losses = []
        for t in range(4):
            state = wm.step(state, rows["action"][t], context)
            state = torch.where(rows["valid"][t, :, None].bool(), state, rows["next"][t])
            physical = wm.spec.normalized_sq_error(state[:, :37], rows["next"][t, :, :37])
            if adaptive:
                physical = (wm.spec.error_vector(state[:, :37], rows["next"][t, :, :37])*
                            fixed_scale/error_norm.std().clamp_min(floor)).square().mean(-1)
            contact = contact_bce(state[:, 37:], rows["next"][t, :, 37:])
            losses.append(physical + contact)
        loss = (torch.stack(losses)*rows["valid"]).sum()/rows["valid"].sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    return dict(seconds=time.monotonic()-started, **evaluate(wm, validation))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-xml", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    args = p.parse_args()
    if args.steps < 16 or args.updates < 1:
        p.error("steps >=16 and updates >=1 required")
    torch.set_num_threads(1)
    xml = resolve_robot_xml(Path("."), args.robot_xml)
    report = dict(protocol="8 real worlds, first 4 fit / last 4 held out; paired data/initialization/windows",
                  steps=args.steps, updates=args.updates, results=[])
    for seed in args.seeds:
        train, valid = collect(xml, args.steps, seed)
        row = dict(seed=seed, legacy=fit(train, valid, args.updates, seed, True),
                   corrected=fit(train, valid, args.updates, seed, False),
                   corrected_scaled=fit(train, valid, args.updates, seed, False, adaptive=True))
        report["results"].append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

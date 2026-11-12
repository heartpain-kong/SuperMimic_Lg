"""Independent learner batches, fractional collection and resumed update accounting."""
import json
import os
from pathlib import Path
import pytest
import torch

from supermimic_lg.cli import parser
from supermimic_lg.runners.velocity import VelocityTrainer


def test_real_physics_utd_and_learning_counters_resume(tmp_path, monkeypatch):
    xml = os.environ.get("LGDOG_TEST_XML")
    if not xml or not Path(xml).is_file():
        pytest.skip("set LGDOG_TEST_XML for actual LgDog physics")
    args = parser().parse_args(["--task", "velocity", "--robot-xml", xml, "--backend", "mujoco",
        "--device", "cpu", "--num-envs", "2", "--iterations", "2", "--collect-steps", "17",
        "--update-to-data-ratio", ".4", "--batch-size", "99", "--wm-batch-size", "3",
        "--policy-batch-size", "5", "--wm-updates", "1", "--policy-updates", "1",
        "--warmup-iterations", "1", "--horizon", "2", "--wm-horizon", "2",
        "--critic-epochs", "1", "--critic-minibatches", "3", "--wm-hidden", "16,16",
        "--actor-hidden", "16,16", "--critic-hidden", "16,16", "--output-dir", str(tmp_path/'run')])
    trainer = VelocityTrainer(args)
    sampled = []
    original = trainer.replay.sample

    def sample(batch, horizon=1, **kwargs):
        sampled.append((batch, horizon))
        return original(batch, horizon, **kwargs)

    monkeypatch.setattr(trainer.replay, 'sample', sample)
    trainer.run()
    assert sampled == [(3, 2), (5, 1)]*2
    assert trainer.transitions == 10
    assert trainer.learner.counts == dict(wm=2, actor=2, critic=6)
    checkpoint = torch.load(tmp_path/'run/checkpoint.pt', weights_only=True)
    assert checkpoint['learner']['updates'] == trainer.learner.counts
    assert checkpoint['metadata']['learning']['wm_batch_size'] == 3
    args.resume = tmp_path/'run/checkpoint.pt'; args.iterations = 3
    resumed = VelocityTrainer(args)
    assert resumed.learner.counts == trainer.learner.counts
    resumed.run()
    assert resumed.transitions == 14
    assert resumed.learner.counts == dict(wm=3, actor=3, critic=9)
    metrics = [json.loads(line) for line in (tmp_path/'run/metrics.jsonl').read_text().splitlines()]
    assert [row['collection_steps'] for row in metrics] == [2, 3, 2]
    assert metrics[-1]['updates/actor'] == 3
    args.policy_batch_size = 6; args.iterations = 4
    resized = VelocityTrainer(args)
    assert resized.learning_cfg.policy_batch_size == 6
    assert resized.learner.counts == resumed.learner.counts
    resized.sim.close()
    args.gamma = .9
    with pytest.raises(ValueError, match='contract differs'):
        VelocityTrainer(args)

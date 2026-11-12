"""Contact losses must leave autocast in collection and differentiable fitting."""
import json
import math
import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_contact_loss_fp32_is_finite_at_probability_boundaries(dtype, monkeypatch):
    from supermimic_lg.learning.losses import contact_bce
    probabilities = torch.tensor([[0., 1., .25, .75, .5]], dtype=dtype, requires_grad=True)
    targets = torch.tensor([[1., 0., 0., 1., 1.]], dtype=dtype)
    expected = F.binary_cross_entropy(probabilities.detach().float().clamp(1e-6, 1-1e-6),
                                     targets.float(), reduction='none').mean(-1)
    original_bce = F.binary_cross_entropy

    def strict_bce(inputs, labels, *args, **kwargs):
        assert not torch.is_autocast_enabled('cpu')
        assert inputs.dtype == labels.dtype == torch.float32
        return original_bce(inputs, labels, *args, **kwargs)

    monkeypatch.setattr(F, 'binary_cross_entropy', strict_bce)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = contact_bce(probabilities, targets)
        assert torch.is_autocast_enabled('cpu'), 'restore the caller autocast context'
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    loss.sum().backward()
    assert torch.isfinite(probabilities.grad).all() and probabilities.grad[:, 2:].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA for native autocast prohibition')
def test_contact_loss_cuda_bf16_backward():
    from supermimic_lg.learning.losses import contact_bce
    model = torch.nn.Linear(4, 5).cuda()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        probabilities = model(torch.randn(3, 4, device='cuda')).float().sigmoid()
        loss = contact_bce(probabilities, torch.ones_like(probabilities)).mean()
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('algorithm', ['supermimic', 'xmimic'])
def test_velocity_bf16_collection_and_learning_disable_bce_autocast(tmp_path, monkeypatch, algorithm):
    xml = os.environ.get('LGDOG_TEST_XML')
    if not xml or not Path(xml).is_file():
        pytest.skip('set LGDOG_TEST_XML for real LgDog physics')
    from supermimic_lg.cli import parser
    from supermimic_lg.runners.velocity import VelocityTrainer

    args = parser().parse_args(['--algorithm', algorithm, '--task', 'velocity', '--robot-xml', xml,
        '--backend', 'mujoco', '--device', 'cpu', '--num-envs', '2', '--iterations', '2',
        '--collect-steps', '4', '--wm-horizon', '2', '--horizon', '3', '--batch-size', '4',
        '--wm-hidden', '16,16,16', '--actor-hidden', '16,16,16', '--critic-hidden', '16,16',
        '--wm-updates', '1', '--policy-updates', '1', '--warmup-iterations', '1',
        '--critic-epochs', '1', '--critic-minibatches', '2', '--torch-threads', '1',
        '--output-dir', str(tmp_path/algorithm)] + (['--warmup-steps', '4'] if algorithm == 'xmimic' else []))
    trainer = VelocityTrainer(args)
    # CPU permits BCE under autocast. Enforce CUDA's prohibition to reproduce the
    # reported failure on machines without a CUDA runtime, using real BF16 MLPs.
    monkeypatch.setattr(trainer, 'autocast', lambda: torch.autocast('cpu', dtype=torch.bfloat16))
    original_bce, grad_modes = F.binary_cross_entropy, []

    def strict_bce(inputs, targets, *args, **kwargs):
        assert not torch.is_autocast_enabled(inputs.device.type), 'BCE is unsafe inside autocast'
        assert inputs.dtype == targets.dtype == torch.float32
        grad_modes.append(torch.is_grad_enabled())
        return original_bce(inputs, targets, *args, **kwargs)

    monkeypatch.setattr(F, 'binary_cross_entropy', strict_bce)
    trainer.run()
    assert set(grad_modes) == {False, True}, 'cover both collection diagnostics and WM gradients'
    assert trainer.learner.counts['wm'] == trainer.learner.counts['actor'] == 2
    metrics = [json.loads(line) for line in (tmp_path/algorithm/'metrics.jsonl').read_text().splitlines()]
    assert all(math.isfinite(v) for row in metrics for v in row.values() if isinstance(v, (int, float)))
    checkpoint = torch.load(tmp_path/algorithm/'checkpoint.pt', weights_only=True)
    if algorithm == 'xmimic':
        assert checkpoint['learner']['ema']['updates'] == 2

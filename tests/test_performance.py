"""Numerical/gradient contracts for the batched performance implementation."""
import copy
import os
from pathlib import Path
import pytest
import torch

from supermimic_lg.state import quat_apply, quat_apply_inverse, quat_mul
from supermimic_lg.learning.profiling import LearnerTimings
from supermimic_lg.xmimic_lg.optim import MuonAdam
from supermimic_lg.xmimic_lg.ema import PolicyEMA


@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('shapes', [((4,), (2, 5, 3)), ((2, 1, 4), (2, 5, 3))])
def test_quaternion_vector_formula_matches_hamilton_products_and_gradients(inverse, shapes):
    torch.manual_seed(21)
    # Deliberately non-unit quaternions: the optimized formula must preserve
    # q*[0,v]*conjugate(q), rather than silently assuming norm(q)==1.
    q = torch.randn(shapes[0], dtype=torch.float64, requires_grad=True)
    v = torch.randn(shapes[1], dtype=torch.float64, requires_grad=True)
    reference_q = q*q.new_tensor([1., -1., -1., -1.]) if inverse else q
    reference = quat_mul(quat_mul(reference_q, torch.cat((torch.zeros_like(v[..., :1]), v), -1)),
                         reference_q*reference_q.new_tensor([1., -1., -1., -1.]))[..., 1:]
    actual = (quat_apply_inverse if inverse else quat_apply)(q, v)
    weight = torch.randn_like(actual)
    ref_grad = torch.autograd.grad((reference*weight).sum(), (q, v), retain_graph=True)
    act_grad = torch.autograd.grad((actual*weight).sum(), (q, v))
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    for a,b in zip(act_grad, ref_grad):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_batched_fk_multidimensional_state_has_correct_derivative():
    xml = os.environ.get('LGDOG_TEST_XML')
    if not xml or not Path(xml).is_file():
        pytest.skip('set LGDOG_TEST_XML for LgDog kinematics')
    from supermimic_lg.kinematics import LgDogKinematics
    from supermimic_lg.simulator import load_lgdog_model
    torch.manual_seed(3)
    fk = LgDogKinematics(load_lgdog_model(xml))
    assert len(fk.levels) == 3 and all(len(level[0]) == 4 for level in fk.levels)
    state = (torch.randn(2, 1, 37, dtype=torch.float64)*.1).requires_grad_()
    assert fk(state)[0].shape == (2, 1, 13, 3)
    assert torch.autograd.gradcheck(fk, (state,), fast_mode=True, atol=2e-5, rtol=2e-4)


def test_foreach_adam_handles_missing_gradients_and_old_per_leaf_steps():
    torch.manual_seed(7)
    a = torch.nn.Linear(3, 2)
    b = copy.deepcopy(a)
    optimized, reference = MuonAdam(a, lr=.001), torch.optim.Adam(b.parameters(), lr=.001)
    for step in range(6):
        for index, (pa,pb) in enumerate(zip(a.parameters(), b.parameters())):
            grad = torch.randn_like(pa) if index == 0 or step % 2 else None
            pa.grad = grad
            pb.grad = None if grad is None else grad.clone()
        optimized.step(); reference.step()
    for pa,pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa,pb,atol=1e-7,rtol=1e-6)
    saved = copy.deepcopy(optimized.state_dict())
    c = copy.deepcopy(a); restored = MuonAdam(c, lr=.001); restored.load_state_dict(saved)
    for pa,pc in zip(a.parameters(), c.parameters()):
        pa.grad = torch.randn_like(pa); pc.grad = pa.grad.clone()
    optimized.step(); restored.step()
    for pa,pc in zip(a.parameters(), c.parameters()):
        torch.testing.assert_close(pa,pc,atol=0,rtol=0)


def test_foreach_ema_copies_integer_buffers_and_averages_float_buffers():
    model = torch.nn.Linear(2, 2)
    model.register_buffer('counter', torch.tensor(0, dtype=torch.long))
    model.register_buffer('stat', torch.zeros(2, dtype=torch.float64))
    ema = PolicyEMA(model, decay=.5)
    ema.update(model)
    model.counter.fill_(7); model.stat.fill_(2.)
    ema.update(model)
    assert ema.model.counter.item() == 7
    torch.testing.assert_close(ema.model.stat, torch.ones(2, dtype=torch.float64))


def test_stage_metrics_sum_repeated_updates_and_drain():
    timer = LearnerTimings('cpu')
    for _ in range(2):
        with timer.measure('critic_fit'):
            torch.ones(10).sum()
    metrics = timer.metrics()
    assert metrics['time/critic_fit_s'] > 0 and metrics['time/actor_rollout_s'] == 0.
    assert all(value == 0 for value in timer.metrics().values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA events and multi-tensor kernels')
def test_cuda_stage_timing_foreach_optimizer_and_ema():
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.SiLU(),
        torch.nn.Linear(16, 16), torch.nn.SiLU(), torch.nn.Linear(16, 4)).cuda()
    optimizer = MuonAdam(model)
    ema = PolicyEMA(model)
    timer = LearnerTimings('cuda')
    with timer.measure('actor_backward'):
        model(torch.randn(64, 8, device='cuda')).square().mean().backward()
        optimizer.step()
    with timer.measure('policy_ema'):
        ema.update(model)
    metrics = timer.metrics()
    assert metrics['time/actor_backward_s'] > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert ema.updates == 1

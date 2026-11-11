import torch
import pytest

from supermimic_lg.norm import RunningNorm, normed_mlp
from supermimic_lg.state import QuadrupedStateSpec
from supermimic_lg.velocity_task import VelocityPolicy, VelocityCritic, VelocityWorldModel, frame_dim


def test_running_norm_tracks_batch_statistics_and_freezes():
    torch.manual_seed(0)
    norm = RunningNorm(4, eps=1e-2)
    data = 3. + 2.*torch.randn(512, 4)
    norm.update(data)
    normalized = norm.apply(data)
    assert normalized.mean(0).abs().max() < 0.1
    assert (normalized.std(0) - 1).abs().max() < 0.1
    count = float(norm.count)
    assert count == 512
    norm.update(data, max_count=100.)   # already past the cap: stays frozen
    assert float(norm.count) == count
    # Streaming in two halves matches one batch update.
    split = RunningNorm(4, eps=1e-2)
    split.update(data[:256]); split.update(data[256:])
    torch.testing.assert_close(norm.mean, split.mean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(norm.var, split.var, rtol=1e-5, atol=1e-5)


def test_normed_mlp_is_layered_and_zero_init_holds():
    net = normed_mlp(6, (16, 8), 3, zero_last=True)
    x = torch.randn(5, 6)
    assert torch.allclose(net(x), torch.zeros(5, 3))
    assert any(isinstance(m, torch.nn.LayerNorm) for m in net)
    net[-1].weight.data.fill_(.1)
    assert not torch.allclose(net(x), torch.zeros(5, 3))


def test_delta_between_inverts_integration():
    spec = QuadrupedStateSpec(12, dt=.02)
    torch.manual_seed(1)
    state = torch.zeros(4, 37)
    state[:, 2] = .3
    state[:, 3] = 1.
    state[:, 7:19] = .1*torch.randn(4, 12)
    state[:, 19:22] = .4*torch.randn(4, 3)
    state[:, 22:25] = .2*torch.randn(4, 3)
    state[:, 25:37] = .5*torch.randn(4, 12)
    delta = .05*torch.randn(4, spec.delta_dim)
    moved = spec.integrate(state, delta)
    recovered = spec.delta_between(state, moved)
    torch.testing.assert_close(recovered, delta, rtol=1e-4, atol=1e-4)


def test_world_model_fits_its_normalisers_on_real_transitions():
    torch.manual_seed(0)
    wm = VelocityWorldModel((16, 16))
    state = torch.zeros(8, 42)
    state[:, 2] = .3
    state[:, 3] = 1.
    action = .2*torch.randn(8, 12)
    context = torch.zeros(8, 85)
    nxt = state.clone()
    nxt[:, 2] += .01
    nxt[:, 19:22] = .3
    assert float(wm.in_norm.count) == 0
    wm.update_norms(state, action, context, nxt)
    assert float(wm.in_norm.count) == 8 and float(wm.out_norm.count) == 8
    assert wm.out_norm.std().min() > 0
    predicted = wm.step(state, action, context)
    assert predicted.shape == (8, 42) and torch.isfinite(predicted).all()


def test_policy_head_is_small_linear_and_not_saturating():
    policy = VelocityPolicy(history=3, hidden=(32, 32))
    obs = torch.randn(64, 3, frame_dim(0))
    out = policy(obs)
    assert out.shape == (64, 12)
    assert out.std() < .05            # xmimic scales the head by 0.01
    assert torch.isfinite(out).all()
    # A linear head can leave [-1, 1]; a tanh head could not.
    with torch.no_grad():
        policy.net[-1].bias.fill_(3.)
    assert bool(policy(obs).max() > 1.)
    with torch.no_grad():
        policy.net[-1].bias.zero_()
    loss = policy(obs).square().mean()
    loss.backward()
    assert policy.net[-1].weight.grad is not None and policy.net[-1].weight.grad.abs().sum() > 0


def test_critic_normalises_its_input_and_returns_scalars():
    critic = VelocityCritic(hidden=(16, 16))
    features = 5.*torch.randn(32, 37+5+1+81+5) + 10.
    values = critic(features)
    assert values.shape == (32,)
    critic.norm.update(features)
    assert float(critic.norm.count) == 32
    assert critic.norm.apply(features).mean(0).abs().max() < .5

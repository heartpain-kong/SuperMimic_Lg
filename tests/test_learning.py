"""Numerical and gradient contracts for the simulator-independent learning core."""
from dataclasses import replace
import pytest
import torch

from supermimic_lg.learning.returns import discounted_objective, lambda_returns, survival_weights
from supermimic_lg.learning.optim import frozen_parameters
from supermimic_lg.learning.schedule import CollectionSchedule
from supermimic_lg.learning.velocity import VelocityLearner, VelocityLearningConfig
from supermimic_lg.tasks.velocity.task import VelocityTask
from supermimic_lg.tasks.velocity.networks import VelocityWorldModel, VelocityPolicy, VelocityCritic
from supermimic_lg.tasks.velocity.mdp import frame, phase_at


def test_soft_return_preserves_termination_and_bootstrap_gradients():
    rewards = torch.tensor([[2.], [3.]], requires_grad=True)
    done = torch.tensor([[.2], [1.]], requires_grad=True)
    value = torch.tensor([10.], requires_grad=True)
    objective = discounted_objective(rewards, done, value, gamma=.5)
    torch.testing.assert_close(objective, torch.tensor([3.2]))
    objective.sum().backward()
    torch.testing.assert_close(rewards.grad, torch.tensor([[1.], [.4]]))
    torch.testing.assert_close(done.grad, torch.tensor([[-1.5], [-2.]]))
    assert value.grad.item() == 0.
    targets = lambda_returns(rewards, done, torch.tensor([[4.], [5.], [10.]]), gamma=.5, lam=.95)
    torch.testing.assert_close(targets, torch.tensor([[3.24], [3.]]))
    assert not targets.requires_grad


def test_gae_keeps_local_continuation_after_survival_underflows():
    done = torch.full((256, 2), .9)
    assert survival_weights(done)[150:].eq(0).all()
    targets = lambda_returns(torch.ones_like(done), done, torch.full((257, 2), 2.), lam=1.)
    torch.testing.assert_close(targets[150], torch.full((2,), 1./(1.-.99*.1)))


def test_freeze_restores_original_flags_on_error_and_keeps_input_gradient():
    model = torch.nn.Linear(2, 1)
    model.bias.requires_grad_(False)
    flags = [p.requires_grad for p in model.parameters()]
    x = torch.ones(3, 2, requires_grad=True)
    with pytest.raises(RuntimeError, match="injected"):
        with frozen_parameters(model):
            model(x).sum().backward()
            assert x.grad.abs().sum() > 0
            assert all(p.grad is None for p in model.parameters())
            raise RuntimeError("injected")
    assert [p.requires_grad for p in model.parameters()] == flags


def test_utd_schedule_is_fractional_and_resumes_at_global_iteration():
    cfg = CollectionSchedule(16, .4)
    assert [cfg.steps(i) for i in range(6)] == [2, 3, 2, 3, 2, 3]
    assert sum(cfg.steps(i) for i in range(100)) == 250
    assert sum(cfg.steps(i) for i in range(37, 100)) == 250-sum(cfg.steps(i) for i in range(37))
    assert CollectionSchedule(16).steps(100) == 16
    for ratio in (0., -1., 2., float('nan')):
        with pytest.raises(ValueError):
            CollectionSchedule(16, ratio)


def make_learner(batch=5):
    torch.manual_seed(4)
    wm, policy, critic = VelocityWorldModel((16, 16)), VelocityPolicy(2, (16, 16)), VelocityCritic((16, 16))

    def feet(state):
        return (state[:, None, :3]+state.new_tensor([0., 0., -.28])).expand(-1, 4, -1), state[:, None, 19:22].expand(-1, 4, -1)

    cfg = VelocityLearningConfig(wm_batch_size=batch, policy_batch_size=batch,
        horizon=3, wm_horizon=2, critic_epochs=2, critic_minibatches=4, value_target_limit=0)
    learner = VelocityLearner(wm, policy, critic, VelocityTask(feet), cfg)
    state = torch.zeros(batch, 42)
    state[:, 2], state[:, 3] = .3, 1.
    context, previous = torch.zeros(batch, 85), torch.zeros(batch, 12)
    command = torch.zeros(batch, 3); command[:, 0] = torch.linspace(.1, .5, batch)
    phase = phase_at(torch.zeros(batch))
    history = frame(state, previous, command, phase)[:, None].repeat(1, 2, 1)
    rows = {key: value[None] for key, value in dict(state=state, context=context, history=history,
        previous=previous, command=command, phase=phase).items()}
    return learner, rows


def test_learner_targets_are_fixed_and_critic_minibatches_cover_remainder(monkeypatch):
    learner, rows = make_learner()
    policy_before = {key: value.clone() for key, value in learner.policy.named_parameters()}
    model_before = {key: value.clone() for key, value in learner.wm.state_dict().items()}
    batches, targets_seen = [], []
    import supermimic_lg.learning.velocity as module
    original_returns, original_loss = module.lambda_returns, module.F.mse_loss

    def returns(*args, **kwargs):
        result = original_returns(*args, **kwargs)
        targets_seen.append(result.detach().flatten().clone())
        return result

    def loss(prediction, target):
        batches.append(target.clone())
        return original_loss(prediction, target)

    monkeypatch.setattr(module, 'lambda_returns', returns)
    monkeypatch.setattr(module.F, 'mse_loss', loss)
    metrics = learner.update_policy(rows)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert learner.counts == dict(wm=0, actor=1, critic=8)
    assert any(not torch.equal(value, policy_before[key]) for key, value in learner.policy.named_parameters())
    for key, value in learner.wm.state_dict().items():
        torch.testing.assert_close(value, model_before[key])
    assert float(learner.critic.norm.count) == 4*5  # H+1, including the bootstrap state.
    assert len(targets_seen) == 1
    for epoch in range(2):
        torch.testing.assert_close(torch.cat(batches[epoch*4:epoch*4+4]).sort().values,
                                   targets_seen[0].sort().values)
    saved = learner.state_dict()
    other, _ = make_learner()
    other.load_state_dict(saved)
    assert other.counts == learner.counts
    other.cfg = replace(other.cfg, gamma=.9)
    with pytest.raises(ValueError, match='configuration differs'):
        other.load_state_dict(saved)


def test_failed_rollout_does_not_leave_learner_frozen(monkeypatch):
    learner, rows = make_learner()
    learner.wm.contacts[-1].bias.requires_grad_(False)
    flags = [[p.requires_grad for p in m.parameters()] for m in (learner.wm, learner.critic)]
    monkeypatch.setattr(learner.task, 'reward', lambda *args: (_ for _ in ()).throw(RuntimeError('injected')))
    with pytest.raises(RuntimeError, match='injected'):
        learner.update_policy(rows)
    assert flags == [[p.requires_grad for p in m.parameters()] for m in (learner.wm, learner.critic)]
    assert learner.counts == dict(wm=0, actor=0, critic=0)

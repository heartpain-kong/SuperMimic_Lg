"""Full-horizon batched reward/feature reuse and AMP update correctness."""
import os
import warnings
from pathlib import Path
import pytest
import torch
from supermimic_lg.learning.optim import optimizer_step, active_autocast_dtype
from supermimic_lg.learning.rollout import imagine
from supermimic_lg.learning.returns import discounted_objective
from supermimic_lg.learning.optim import frozen_parameters
from supermimic_lg.tasks.velocity.imagination import imagine_batched, critic_features, CompiledRecurrentStep
from supermimic_lg.tasks.velocity.networks import VelocityWorldModel, VelocityPolicy, VelocityCritic
from supermimic_lg.tasks.velocity.task import VelocityTask, VelocityCondition


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA for native AMP cache regression'))])
def test_optimizer_update_invalidates_live_autocast_weight_cache(device):
    model = torch.nn.Linear(2, 2, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    x = torch.ones(1, 2, device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        before = model(x)
        optimizer_step(before.square().mean(), model, optimizer, grad_clip=100.)
        actual = model(x).detach()
        expected = torch.nn.functional.linear(x.bfloat16(), model.weight.detach().bfloat16())
    assert not torch.equal(actual, before.detach()), 'optimizer step must change the following forward'
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def fixture(history=4):
    xml = os.environ.get('LGDOG_TEST_XML')
    if not xml or not Path(xml).is_file():
        pytest.skip('set LGDOG_TEST_XML for real LgDog geometry')
    from supermimic_lg.tasks.velocity.mdp import FootKinematics
    from supermimic_lg.velocity_physics import offsets
    from supermimic_lg.config import lgdog1_spec
    torch.manual_seed(31)
    wm, actor, critic = VelocityWorldModel((16,16)), VelocityPolicy(history,(16,16)), VelocityCritic((16,16))
    with torch.no_grad():
        wm.dynamics[-1].weight.normal_(0, 1e-4)  # nonzero action-to-state Jacobian
        actor.net[-1].bias[0] = 1.3  # also exercise clamp/bound-penalty gradients
        actor.obs_norm.mean.normal_(0, .1)
        actor.obs_norm.var.uniform_(.5, 1.5)
    state = torch.randn(3,42)*.02
    state[:,2], state[:,3] = .31, 1.
    state[:,3:7] /= state[:,3:7].norm(dim=-1,keepdim=True)
    state[:,7:19] += torch.tensor(lgdog1_spec().default_joint_pos)
    state[:,37:] = .2
    context = torch.zeros(3,85)
    context[:,:2], context[:,3] = state[:,:2], .3
    grid = offsets()
    context[:,4:] = grid[:,0]*.1+grid[:,1]*.05
    previous = torch.randn(3,12)*.1
    command = torch.tensor([[.4,.1,.2]]*3)
    phase = torch.tensor([[.6,.8]]*3)
    observation = torch.randn(3,history,47)*.1
    tensors = [x.requires_grad_() for x in (state, context, observation, previous, command, phase)]
    return wm, actor, critic, VelocityTask(FootKinematics(xml,'cpu')), tensors


@pytest.mark.parametrize('horizon', [1,8,32])
@pytest.mark.parametrize('history', [1,4])
def test_batched_rollout_matches_rewards_bootstrap_and_all_input_policy_gradients(horizon, history):
    wm, actor, critic, task, inputs = fixture(history)
    state, context, observations, previous, command, phase = inputs
    condition = VelocityCondition(command, phase)
    with frozen_parameters(wm, critic):
        ref = imagine(wm, actor, task, state, context, observations, previous, condition, horizon)
        fast = imagine_batched(wm, actor, task, state, context, observations, previous, condition, horizon)
        for name in ('states','actions','rewards','termination','survival','action_bounds'):
            torch.testing.assert_close(getattr(fast,name),getattr(ref,name),atol=2e-6,rtol=2e-5)
        xref, xfast = [critic_features(r,task,wm,context) for r in (ref,fast)]
        torch.testing.assert_close(xfast,xref,atol=2e-6,rtol=2e-5)
        objectives = [discounted_objective(r.rewards,r.termination,critic(x[-1])).mean()
                      -r.action_bounds.mean() for r,x in ((ref,xref),(fast,xfast))]
        leaves = inputs+list(actor.parameters())
        ga = torch.autograd.grad(objectives[0], leaves, retain_graph=True)
        gb = torch.autograd.grad(objectives[1], leaves)
        for a,b in zip(ga,gb):
            torch.testing.assert_close(a,b,atol=5e-5,rtol=3e-4)


def test_critic_normalizes_once_after_its_stats_update():
    from supermimic_lg.learning.velocity import VelocityLearner, VelocityLearningConfig
    wm, actor, critic, task, tensors = fixture()
    state, context, history, previous, command, phase = [v.detach() for v in tensors]
    learner = VelocityLearner(wm,actor,critic,task,VelocityLearningConfig(horizon=3,critic_epochs=2,critic_minibatches=4))
    rows = {k:v[None] for k,v in dict(state=state,context=context,history=history,
                                     previous=previous,command=command,phase=phase).items()}
    calls = []
    original = critic.norm.apply
    critic.norm.apply = lambda x: (calls.append(x.shape), original(x))[1]
    learner.update_policy(rows)
    assert len(calls) == 2  # actor bootstrap plus full H+1 target/regression batch
    assert learner.counts == dict(wm=0,actor=1,critic=8)


def test_bf16_batched_rollout_matches_reference_gradients():
    wm, actor, critic, task, inputs = fixture()
    state, context, history, previous, command, phase = inputs
    condition = VelocityCondition(command,phase)
    with frozen_parameters(wm,critic), torch.autocast('cpu',dtype=torch.bfloat16):
        ref = imagine(wm,actor,task,state,context,history,previous,condition,8)
        fast = imagine_batched(wm,actor,task,state,context,history,previous,condition,8)
        torch.testing.assert_close(fast.rewards,ref.rewards,atol=2e-5,rtol=2e-4)
        objectives = [discounted_objective(r.rewards,r.termination,
            critic(critic_features(r,task,wm,context,last_only=True)[0])).mean()-r.action_bounds.mean()
            for r in (ref,fast)]
        leaves = inputs+list(actor.parameters())
        a = torch.autograd.grad(objectives[0],leaves,retain_graph=True)
        b = torch.autograd.grad(objectives[1],leaves)
    for ga,gb in zip(a,b):
        torch.testing.assert_close(ga,gb,atol=1e-4,rtol=2e-3)


@pytest.mark.parametrize('precision', ['fp32','bf16'])
def test_compiled_recurrent_step_fullgraph_forward_and_backward(precision):
    wm, actor, critic, task, inputs = fixture()
    state, context, history, previous, command, phase = inputs
    condition = VelocityCondition(command,phase)
    with frozen_parameters(wm,critic), torch.autocast('cpu',dtype=torch.bfloat16,enabled=precision=='bf16'):
        eager = imagine_batched(wm,actor,task,state,context,history,previous,condition,3)
        # Test Dynamo/AOT full-graph capture without requiring CUDA/Triton or a
        # system C++ compiler. This is a correctness test, not an Inductor speed claim.
        compiled = CompiledRecurrentStep(wm,actor,task,backend='aot_eager')
        traced = imagine_batched(wm,actor,task,state,context,history,previous,condition,3,step_fn=compiled)
        torch.testing.assert_close(traced.states,eager.states,atol=2e-6,rtol=2e-5)
        losses = [r.rewards.mean()-r.action_bounds.mean() for r in (eager,traced)]
        params = list(actor.parameters())
        a = torch.autograd.grad(losses[0],params,retain_graph=True)
        b = torch.autograd.grad(losses[1],params)
    for ga,gb in zip(a,b):
        torch.testing.assert_close(ga,gb,atol=1e-4 if precision=='bf16' else 5e-5,
                                   rtol=2e-3 if precision=='bf16' else 3e-4)


def test_bf16_critic_reuses_quantized_fixed_inputs():
    from supermimic_lg.learning.velocity import VelocityLearner, VelocityLearningConfig
    wm, actor, critic, task, tensors = fixture()
    state, context, history, previous, command, phase = [v.detach() for v in tensors]
    learner = VelocityLearner(wm,actor,critic,task,VelocityLearningConfig(horizon=3,critic_epochs=2,critic_minibatches=4))
    rows = {k:v[None] for k,v in dict(state=state,context=context,history=history,
                                     previous=previous,command=command,phase=phase).items()}
    dtypes = []
    hook = critic.net[0].register_forward_pre_hook(lambda m,args: dtypes.append(args[0].dtype))
    try:
        with torch.autocast('cpu',dtype=torch.bfloat16):
            learner.update_policy(rows)
    finally:
        hook.remove()
    assert dtypes == [torch.float32]+[torch.bfloat16]*9  # bootstrap, targets, 8 fits


def test_amp_dtype_query_keeps_torch21_compatibility(monkeypatch):
    assert active_autocast_dtype('cpu') is None
    with torch.autocast('cpu',dtype=torch.bfloat16):
        assert active_autocast_dtype('cpu') == torch.bfloat16
        with monkeypatch.context() as patch, warnings.catch_warnings():
            patch.delattr(torch,'get_autocast_dtype',raising=False)
            warnings.simplefilter('ignore', FutureWarning)
            assert active_autocast_dtype('cpu') == torch.bfloat16

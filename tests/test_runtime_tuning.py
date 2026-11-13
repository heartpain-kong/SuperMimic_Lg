import copy
import random
import numpy as np
import pytest
import torch
from supermimic_lg.learning.runtime_tuning import tune_rollout
from supermimic_lg.xmimic_lg.learner import XMimicLearner, XMimicLearningConfig
from supermimic_lg.tasks.velocity.networks import VelocityWorldModel,VelocityPolicy,VelocityCritic
from supermimic_lg.tasks.velocity.task import VelocityTask
from supermimic_lg.tasks.velocity.mdp import frame,phase_at


def make_case():
    torch.manual_seed(5)
    wm,actor,critic = VelocityWorldModel((16,16)),VelocityPolicy(2,(16,16)),VelocityCritic((16,16))
    def feet(s):
        return (s[:,None,:3]+s.new_tensor([0.,0.,-.28])).expand(-1,4,-1),s[:,None,19:22].expand(-1,4,-1)
    learner = XMimicLearner(wm,actor,critic,VelocityTask(feet),XMimicLearningConfig(
        horizon=2,critic_epochs=1,critic_minibatches=2))
    state = torch.zeros(3,42);state[:,2],state[:,3]=.3,1.
    previous,command,context = torch.zeros(3,12),torch.tensor([[.4,0.,0.]]*3),torch.zeros(3,85)
    phase = phase_at(torch.zeros(3))
    history = frame(state,previous,command,phase)[:,None].repeat(1,2,1)
    rows = {k:v[None] for k,v in dict(state=state,context=context,previous=previous,
        command=command,phase=phase,history=history).items()}
    learner.update_policy(rows)  # populate moments/EMA before snapshotting
    learner.timings.reset()
    return learner,rows


def state_of(learner):
    return copy.deepcopy(dict(models=[m.state_dict() for m in (learner.wm,learner.policy,learner.critic)],
        optimizers=[o.state_dict() for o in learner.optimizers],learner=learner.state_dict(),
        grads=[p.grad for m in (learner.wm,learner.policy,learner.critic) for p in m.parameters()]))


def equal(a,b):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:
        assert a==b


def test_trials_restore_weights_moments_ema_gradients_and_rng():
    learner,rows=make_case()
    before=state_of(learner)
    rng=torch.get_rng_state(); prng=random.getstate(); nrng=np.random.get_state()
    report=tune_rollout(learner,rows,candidates=('batched','reference'),samples=1)
    equal(before,state_of(learner))
    assert torch.equal(rng,torch.get_rng_state()) and prng==random.getstate()
    assert np.array_equal(nrng[1],np.random.get_state()[1])
    assert report['training_state_restored'] and all(v['status']=='ok' for v in report['candidates'].values())


@pytest.mark.parametrize('interrupt',[False,True])
def test_failed_or_interrupted_trial_restores_training_state(monkeypatch,interrupt):
    learner,rows=make_case();before=state_of(learner)
    original=learner.update_policy
    def update(rows):
        if learner.rollout_mode=='compiled':
            with torch.no_grad():next(learner.policy.parameters()).add_(10.)
            torch.rand(7)
            raise KeyboardInterrupt() if interrupt else RuntimeError('simulated compiler failure')
        return original(rows)
    monkeypatch.setattr(learner,'update_policy',update)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):tune_rollout(learner,rows,samples=1)
    else:
        report=tune_rollout(learner,rows,samples=1)
        assert report['selected']=='batched' and report['candidates']['compiled']['status']=='failed'
    equal(before,state_of(learner))


def test_numerically_wrong_candidate_is_rejected(monkeypatch):
    learner,rows=make_case();before=state_of(learner);original=learner.update_policy
    def update(rows):
        mode=learner.rollout_mode
        learner.rollout_mode='batched'
        metrics=original(rows)
        learner.rollout_mode=mode
        if mode=='compiled':
            with torch.no_grad():next(learner.policy.parameters()).add_(1.)
        return metrics
    monkeypatch.setattr(learner,'update_policy',update)
    report=tune_rollout(learner,rows,samples=1)
    assert report['selected']=='batched' and report['candidates']['compiled']['status']=='failed'
    equal(before,state_of(learner))


def test_cpu_auto_avoids_compilation_and_counts_only_real_update():
    learner,rows=make_case();before=dict(learner.counts);ema=learner.ema.updates
    learner.rollout_mode='auto';learner.update_policy(rows)
    assert learner.rollout_mode=='batched' and learner.counts['actor']==before['actor']+1
    assert learner.ema.updates==ema+1 and learner.runtime_tuning_report['training_state_restored']

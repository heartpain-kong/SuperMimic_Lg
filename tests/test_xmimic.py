"""XMimic adaptation: optimizer equations, averaged policies and physical integration."""
import json
import os
from pathlib import Path
import copy
import pytest
import torch

from supermimic_lg.cli import parser
from supermimic_lg.xmimic_lg.optim import MuonAdam, hidden_matrices
from supermimic_lg.xmimic_lg.ema import PolicyEMA
from supermimic_lg.xmimic_lg.dynamics import XMimicStateSpec
from supermimic_lg.xmimic_lg.mirror import mirror_state, mirror_context, mirror_joints, augment_windows
from supermimic_lg.tasks.velocity.mdp import sample_ground
from supermimic_lg.scripts.xmimic_native import NATIVE_ROOT, source_status, build_command, command_motion_dir


def test_profile_defaults_and_explicit_overrides_preserve_standard_algorithm():
    standard = parser().parse_args([])
    assert standard.algorithm == 'supermimic' and standard.horizon == 8 and standard.warmup_steps == 0
    x = parser().parse_args(['--algorithm','xmimic','--task','velocity'])
    assert (x.horizon,x.wm_horizon,x.wm_updates,x.policy_updates) == (32,8,5,1)
    assert (x.wm_batch_size,x.policy_batch_size,x.update_to_data_ratio,x.warmup_steps) == (512,256,.4,64)
    assert x.normalize_actor_loss is False and x.wm_mirror_fraction == 0.
    overridden = parser().parse_args(['--algorithm=xmimic','--task','velocity','--horizon=5',
        '--batch-size','7','--normalize-actor-loss','--policy-ema-decay','.8'])
    from supermimic_lg.xmimic_lg.learner import XMimicLearningConfig
    cfg = XMimicLearningConfig.from_args(overridden)
    assert cfg.horizon == 5 and cfg.wm_batch_size == cfg.policy_batch_size == 7
    assert cfg.normalize_actor_loss and cfg.policy_ema_decay == .8


def test_muon_selects_only_interior_matrices_and_adam_matches_torch():
    torch.manual_seed(5)
    network = torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.SiLU(),torch.nn.Linear(5,4),
        torch.nn.SiLU(),torch.nn.Linear(4,2))
    assert hidden_matrices(network) == {id(network[2].weight)}
    optimizer = MuonAdam(network, lr=.001)
    for p in network.parameters():
        p.grad = torch.ones_like(p)*.2
    before = network[2].weight.detach().clone()
    optimizer.step()
    assert 'momentum_buffer' in optimizer.state[network[2].weight]
    assert 'exp_avg' in optimizer.state[network[0].weight]
    assert not torch.equal(before,network[2].weight)
    # A single Linear has no interior matrices: every leaf must match torch Adam.
    a = torch.nn.Linear(3,2); b = copy.deepcopy(a)
    oa, ob = MuonAdam(a, lr=.003, betas=(.9,.95)), torch.optim.Adam(b.parameters(),lr=.003,betas=(.9,.95))
    for _ in range(4):
        for pa,pb in zip(a.parameters(),b.parameters()):
            gradient = torch.randn_like(pa); pa.grad=gradient;pb.grad=gradient.clone()
        oa.step(); ob.step()
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa,pb,atol=1e-7,rtol=1e-6)


def test_muon_optimizer_state_resumes_exactly():
    torch.manual_seed(8)
    a = torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Linear(5,4),torch.nn.Linear(4,2))
    oa = MuonAdam(a)
    for p in a.parameters():p.grad=torch.randn_like(p)
    oa.step()
    b=copy.deepcopy(a);ob=MuonAdam(b);ob.load_state_dict(copy.deepcopy(oa.state_dict()))
    for pa,pb in zip(a.parameters(),b.parameters()):
        gradient=torch.randn_like(pa);pa.grad=gradient;pb.grad=gradient.clone()
    oa.step();ob.step()
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,atol=0,rtol=0)


def test_policy_ema_copies_first_then_averages_parameters_and_buffers():
    from supermimic_lg.velocity_task import VelocityPolicy
    policy=VelocityPolicy(2,(8,8));ema=PolicyEMA(policy,.5)
    with torch.no_grad():
        policy.net[-1].bias.fill_(2.);policy.obs_norm.mean.fill_(3.)
    ema.update(policy)
    with torch.no_grad():
        policy.net[-1].bias.fill_(4.);policy.obs_norm.mean.fill_(5.)
    ema.update(policy)
    torch.testing.assert_close(ema.model.net[-1].bias,torch.full((12,),3.))
    torch.testing.assert_close(ema.model.obs_norm.mean,torch.full((47,),4.))
    restored=PolicyEMA(policy,.5);restored.load_state_dict(ema.state_dict())
    assert restored.updates==2
    for key,value in ema.model.state_dict().items():torch.testing.assert_close(restored.model.state_dict()[key],value)


@pytest.mark.parametrize('drift',[True,False])
def test_structured_drift_targets_invert_step(drift):
    torch.manual_seed(7);spec=XMimicStateSpec(12,explicit_integration=drift)
    state=torch.randn(9,37)*.1;state[:,3:7]=torch.tensor([1.,0.,0.,0.])
    delta=torch.randn(9,36)*.02
    after=spec.integrate(state,delta)
    torch.testing.assert_close(spec.delta_between(state,after),delta,atol=1e-6,rtol=1e-5)


def test_mirror_is_involution_and_preserves_reflected_terrain_samples():
    torch.manual_seed(3)
    state=torch.randn(2,3,42);context=torch.randn(2,3,85)
    torch.testing.assert_close(mirror_state(mirror_state(state)),state)
    torch.testing.assert_close(mirror_context(mirror_context(context)),context)
    joints=torch.randn(2,3,12);torch.testing.assert_close(mirror_joints(mirror_joints(joints)),joints)
    patch=context[0];xy=torch.randn(3,5,2)*.1
    mirrored=xy*xy.new_tensor([1.,-1.])
    torch.testing.assert_close(sample_ground(patch,xy),sample_ground(mirror_context(patch),mirrored),atol=1e-6,rtol=1e-5)
    rows=dict(state=state,next=state,action=joints,context=context,valid=torch.ones(2,3))
    augmented=augment_windows(rows,.5)
    torch.testing.assert_close(augmented['state'][:,:1],mirror_state(state[:,:1]))
    torch.testing.assert_close(augmented['state'][:,1:],state[:,1:])
    torch.testing.assert_close(rows['state'],state)


def test_native_source_is_unchanged_and_launcher_preserves_argv(tmp_path):
    assert source_status(NATIVE_ROOT)['verified']
    command=build_command(NATIVE_ROOT,'python',tmp_path,['--','--layout','1GPU','--log_dir','run with spaces'])
    assert command[-2:]==['--motion_dir',str(tmp_path)]
    assert 'run with spaces' in command
    explicit=build_command(NATIVE_ROOT,'python',tmp_path,['--motion_dir','other'])
    assert explicit.count('--motion_dir')==1
    assert command_motion_dir(NATIVE_ROOT,explicit)==(NATIVE_ROOT/'other').resolve()


def test_native_readiness_checks_original_csv_format_without_importing_jax(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import supermimic_lg.scripts.xmimic_native as native
    pinned=dict(line.strip().split('==',1) for line in (NATIVE_ROOT/'requirements.txt').read_text().splitlines()
                if '==' in line and not line.lstrip().startswith('#'))
    result=SimpleNamespace(returncode=0,stdout=json.dumps(dict(versions=pinned,python=[3,11,9],system='Linux')),stderr='')
    monkeypatch.setattr(native.subprocess,'run',lambda *a,**kw:result)
    (tmp_path/'clip.npz').touch()
    assert not native.readiness(NATIVE_ROOT,'python',tmp_path)['motion_data_present']
    (tmp_path/'clip.csv').touch()
    report=native.readiness(NATIVE_ROOT,'python',tmp_path)
    assert report['motion_data_present'] and not report['ready']  # meshes still absent


def test_lgdog_foot_mirror_is_close_but_not_assumed_exact():
    xml=os.environ.get('LGDOG_TEST_XML')
    if not xml or not Path(xml).is_file():pytest.skip('set LGDOG_TEST_XML for LgDog geometry')
    from supermimic_lg.tasks.velocity.mdp import FootKinematics
    from supermimic_lg.config import lgdog1_spec
    torch.manual_seed(5)
    state=torch.randn(12,42)*.1;state[:,2]=.4
    state[:,3:7]/=state[:,3:7].norm(dim=-1,keepdim=True)
    state[:,7:19]+=torch.tensor(lgdog1_spec().default_joint_pos)
    feet=FootKinematics(Path(xml),'cpu')
    pos,vel=feet(state);mirrored_pos,mirrored_vel=feet(mirror_state(state))
    reflection=state.new_tensor([1.,-1.,1.]);perm=[1,0,3,2]
    assert (mirrored_pos-pos[:,perm]*reflection).abs().max()<2e-4
    assert (mirrored_vel-vel[:,perm]*reflection).abs().max()<1e-4


@pytest.mark.parametrize('seed',[0,1,42])
def test_xmimic_real_physics_checkpoint_ema_and_resume(tmp_path,seed):
    xml=os.environ.get('LGDOG_TEST_XML')
    if not xml or not Path(xml).is_file():pytest.skip('set LGDOG_TEST_XML for LgDog physics')
    from supermimic_lg.runners.velocity import VelocityTrainer
    from supermimic_lg.benchmark import CheckpointActor
    args=parser().parse_args(['--algorithm','xmimic','--task','velocity','--robot-xml',xml,
        '--backend','mujoco','--device','cpu','--num-envs','2','--iterations','2','--warmup-steps','4',
        '--wm-horizon','2','--horizon','3','--batch-size','4','--wm-updates','1','--policy-updates','1',
        '--wm-hidden','16,16,16','--actor-hidden','16,16,16','--critic-hidden','16,16',
        '--critic-epochs','1','--critic-minibatches','3','--episode-steps','20','--seed',str(seed),
        '--policy-ema-decay','.8','--wm-mirror-fraction','.5','--output-dir',str(tmp_path/'run')])
    trainer=VelocityTrainer(args);trainer.run()
    assert trainer.transitions==18 and trainer.learner.ema.updates==2
    ck=torch.load(tmp_path/'run/checkpoint.pt',weights_only=True)
    assert ck['algorithm']=='xmimic_velocity_bptt' and ck['deployment_policy']=='ema'
    assert ck['metadata']['training_revision']==6 and ck['learner']['updates']['actor']==2
    assert any('momentum_buffer' in state for opt in ck['optimizers'][:2] for state in opt['state'].values())
    averaged=CheckpointActor(tmp_path/'run/checkpoint.pt','cpu',Path(xml))
    raw=CheckpointActor(tmp_path/'run/checkpoint.pt','cpu',Path(xml),'raw')
    assert averaged.policy_source=='ema' and raw.policy_source=='raw'
    for key,value in ck['policy_ema'].items():torch.testing.assert_close(averaged.policy.state_dict()[key],value)
    args.resume=tmp_path/'run/checkpoint.pt';args.iterations=3
    resumed=VelocityTrainer(args);assert resumed.learner.ema.updates==2
    for key,value in ck['policy_ema'].items():torch.testing.assert_close(resumed.learner.ema.model.state_dict()[key],value)
    resumed.run()
    assert resumed.transitions==30 and resumed.learner.ema.updates==3
    metrics=[json.loads(row) for row in (tmp_path/'run/metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['xmimic/ema_updates']==3

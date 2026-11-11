import os
from pathlib import Path
import pytest
import torch

from supermimic_lg.velocity_physics import offsets
from supermimic_lg.velocity_task import (sample_ground, geometry, failure, VelocityWorldModel,
                                        VelocityReplay, VelocityPolicy, phase_at, frame)


def state_and_patch():
    state=torch.zeros(2,42)
    state[:,2],state[:,3]=.31,1.
    context=torch.zeros(2,85)
    return state,context


def test_height_patch_slopes_and_gradients_and_elevation_invariance():
    state,context=state_and_patch()
    grid=offsets()
    context[:,4:]=grid[:,0]*.2+grid[:,1]*.1
    xy=torch.tensor([[[.2,.4]],[[-.2,.3]]],requires_grad=True)
    height=sample_ground(context,xy)
    torch.testing.assert_close(height,torch.tensor([[.08],[-.01]]),atol=1e-6,rtol=1e-6)
    height.sum().backward()
    torch.testing.assert_close(xy.grad,torch.tensor([[[.2,.1]],[[.2,.1]]]),atol=1e-6,rtol=1e-6)
    wm=VelocityWorldModel((16, 16))
    before=wm.features(state,context)
    translated=state.clone();translated[:,:3]+=torch.tensor([10.,-7.,4.])
    patch=context.clone();patch[:,:3]+=torch.tensor([10.,-7.,4.]);patch[:,4:]+=4.
    torch.testing.assert_close(wm.features(translated,patch),before,atol=1e-5,rtol=1e-5)
    assert not failure(translated,patch).any()
    translated[0,2]=4.05
    translated[1,41]=1.
    assert failure(translated,patch).all()


def test_contact_model_gradients_and_actor_has_no_terrain_input():
    state,context=state_and_patch()
    wm=VelocityWorldModel((16, 16))
    action=torch.zeros(2,12,requires_grad=True)
    predicted=wm.step(state,action,context)
    assert predicted.shape==(2,42)
    predicted[:,37:].sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum()>0
    obs=frame(state,torch.zeros(2,12),torch.tensor([[.5,0.,0.]]*2),phase_at(torch.zeros(2)))
    assert obs.shape==(2,47)
    assert VelocityPolicy(3, (16, 16))(obs[:,None].repeat(1,3,1)).shape==(2,12)


def test_velocity_replay_masks_post_reset_and_preserves_terminal_target():
    state,context=state_and_patch()
    replay=VelocityReplay(4,1,3,"cpu")
    for i in range(2):
        replay.add(state=state[:1]+i,next=state[:1]+10+i,context=context[:1],history=torch.zeros(1,3,47),
            action=torch.zeros(1,12),previous=torch.zeros(1,12),command=torch.zeros(1,3),phase=torch.zeros(1,2),
            terminal=torch.tensor([1.]),reset=torch.tensor([1.]))
    rows=replay.sample(1,2)
    assert rows['valid'][:,0].tolist()==[1.,0.]
    torch.testing.assert_close(rows['next'][0,0],state[0]+10)


def test_training_contract_matches_native_gpu_and_detects_backend_mismatch(tmp_path):
    from types import SimpleNamespace
    from supermimic_lg.velocity_contract import shared_task_contract
    xml=tmp_path/'robot.xml';xml.write_text('<mujoco/>')
    values=dict(terrain='flat',history=3,command_vx=.6,command_vy=.3,command_yaw=.6,
                episode_steps=500,seed=42)
    native=SimpleNamespace(**values)
    gpu=SimpleNamespace(**values,backend='mjlab')
    cpu=SimpleNamespace(**values,backend='mujoco')
    assert shared_task_contract(xml,tmp_path,native)[1]==shared_task_contract(xml,tmp_path,gpu)[1]
    assert shared_task_contract(xml,tmp_path,cpu)[1]!=shared_task_contract(xml,tmp_path,gpu)[1]
    for override in (dict(reward_velocity=3.), dict(swing_height=.1), dict(command_mode='fixed'),
                     dict(standing_fraction=0.)):
        changed=SimpleNamespace(**{**values,**override})
        assert shared_task_contract(xml,tmp_path,changed)[1]!=shared_task_contract(xml,tmp_path,native)[1]


def test_command_sampling_and_invalid_objectives():
    from supermimic_lg.velocity_objective import sample_commands,objective_from,command_settings
    fixed=dict(command_mode='fixed',command_vx=-.4,command_vy=0.,command_yaw=.2,standing_fraction=0.)
    torch.testing.assert_close(sample_commands(fixed,20,'cpu'),torch.tensor([[-.4,0.,.2]]).repeat(20,1))
    assert not sample_commands({**fixed,'standing_fraction':1.},20,'cpu').any()
    random=sample_commands(dict(command_vx=.4,command_vy=0.,command_yaw=.2,standing_fraction=0.),1000,'cpu')
    assert random[:,0].min()<0<random[:,0].max() and random[:,0].abs().max()<=.4
    assert not random[:,1].any() and random[:,2].abs().max()<=.2
    for bad in (dict(reward_velocity=float('nan')),dict(reward_slip=.1),dict(target_height=.05),dict(swing_height=-.1)):
        with pytest.raises(ValueError):objective_from(bad)
    for bad in (dict(command_mode='unknown'),dict(command_vx=-.1),dict(standing_fraction=1.1)):
        with pytest.raises(ValueError):command_settings(bad)


def test_configurable_reward_contributions_and_bptt_gradient():
    from supermimic_lg.velocity_task import reward
    from supermimic_lg.velocity_objective import VelocityObjective,objective_from
    from supermimic_lg.config import lgdog1_spec
    state,context=state_and_patch()
    state[:,7:19]=torch.tensor(lgdog1_spec().default_joint_pos)
    state.requires_grad_()
    action=torch.zeros(2,12)
    command=torch.tensor([[.4,0.,0.]]*2)
    def feet(s):
        pos=(s[:,:3]+s.new_tensor([0.,0.,-.26]))[:,None,:].expand(-1,4,-1)
        return pos,s[:,19:22][:,None,:].expand(-1,4,-1)
    args=(state,action,action,command,phase_at(torch.zeros(2)),context,feet)
    total,terms=reward(*args,return_terms=True)
    torch.testing.assert_close(total,sum(terms.values()))
    cfg=objective_from({**VelocityObjective().as_dict(),'reward_velocity':3.,'swing_height':.10})
    updated,new_terms=reward(*args,objective=cfg,return_terms=True)
    torch.testing.assert_close(new_terms['velocity'],terms['velocity']*1.5)
    assert not torch.allclose(new_terms['lift'],terms['lift'])
    new_terms['velocity'].sum().backward()
    assert torch.isfinite(state.grad).all() and (state.grad[:,19]>0).all()


def test_native_him_shared_task_configuration_and_gpu_ray_order():
    xml=os.environ.get('LGDOG_TEST_XML')
    if not xml or not Path(xml).is_file():pytest.skip('set LGDOG_TEST_XML for native configuration validation')
    pytest.importorskip('mjlab')
    from supermimic_lg.adapters.mjlab_velocity import make_him_task,shared_reward,physical_failure
    from mjlab.scene import Scene
    cfg,agent=make_him_task(Path(xml).parents[5],terrain='stairs_up',n=2,history=3,width=32)
    assert cfg.scale_rewards_by_dt is False and cfg.decimation==4 and cfg.sim.mujoco.timestep==.005
    assert cfg.rewards['velocity_task'].func is shared_reward
    assert cfg.terminations['physical_failure'].func is physical_failure
    assert tuple(cfg.events)==('reset_shared',)
    assert cfg.observations['actor'].history_length==agent.actor.history_length==3
    assert not cfg.observations['actor'].enable_corruption
    assert cfg.commands['twist'].resampling_time_range==(5.,5.)
    assert cfg.commands['twist'].ranges.lin_vel_x==(-.6,.6)
    custom,_=make_him_task(Path(xml).parents[5],command_mode='fixed',command_vx=-.4,
        command_vy=0.,command_yaw=0.,standing_fraction=0.,objective={'reward_velocity':3.,'swing_height':.1})
    assert custom.commands['twist'].ranges.lin_vel_x==(-.4,-.4)
    assert custom.commands['twist'].rel_standing_envs==0.
    assert custom.rewards['velocity_task'].params['objective']['reward_velocity']==3.
    assert custom.rewards['velocity_task'].params['objective']['swing_height']==.1
    scan=cfg.scene.sensors[0]
    rays,directions=scan.pattern.generate_rays(None,'cpu')
    torch.testing.assert_close(rays[:,:2],offsets())
    assert (directions[:,2]==-1).all()
    scene=Scene(cfg.scene,device='cpu')
    model=scene.compile()
    assert model.nq==19 and model.nv==18 and model.nsensor>=5


@pytest.mark.parametrize('seed',[0,1,42])
def test_reference_free_real_physics_training_and_resume(tmp_path,seed):
    xml=os.environ.get('LGDOG_TEST_XML')
    if not xml or not Path(xml).is_file():pytest.skip('set LGDOG_TEST_XML for real LgDog integration')
    from supermimic_lg.scripts.train_supermimic import parser
    from supermimic_lg.scripts.train_velocity import VelocityTrainer
    args=parser().parse_args(['--task','velocity','--robot-xml',xml,'--backend','mujoco','--device','cpu',
        '--num-envs','2','--iterations','2','--collect-steps','4','--wm-updates','1','--policy-updates','1',
        '--warmup-iterations','1','--batch-size','4','--wm-hidden', '16,16', '--actor-hidden', '16,16', '--critic-hidden', '16,16','--horizon','2','--wm-horizon','2',
        '--episode-steps','4','--seed',str(seed),'--output-dir',str(tmp_path/'run'),
        '--command-mode','fixed','--command-vx','.4','--command-vy','0','--command-yaw','0',
        '--standing-fraction','0','--reward-velocity','3','--swing-height','.09'])
    assert args.motion_file is None
    trainer=VelocityTrainer(args)
    trainer.run()
    torch.testing.assert_close(trainer.command,torch.tensor([[.4,0.,0.]]*2))
    ck=torch.load(tmp_path/'run/checkpoint.pt',weights_only=True)
    assert ck['algorithm']=='supermimic_velocity_bptt' and ck['transitions']==16
    assert ck['metadata']['objective']['reward_velocity']==3.
    assert ck['metadata']['command_config']['mode']=='fixed'
    args.resume=tmp_path/'run/checkpoint.pt';args.iterations=3
    args.reward_velocity=4.
    with pytest.raises(ValueError,match='contract differs'):VelocityTrainer(args)
    args.reward_velocity=3.
    resumed=VelocityTrainer(args)
    resumed.run()
    assert resumed.transitions==24 and resumed.iteration==3
    args.resume=None;args.init_policy=tmp_path/'run/checkpoint.pt';args.output_dir=tmp_path/'transfer'
    initialized=VelocityTrainer(args)
    assert initialized.iteration==0 and initialized.transitions==0
    for key,value in resumed.policy.state_dict().items():
        torch.testing.assert_close(initialized.policy.state_dict()[key],value)
    initialized.sim.close()

def test_oom_hint_halves_memory_dominant_sizes():
    from types import SimpleNamespace
    from supermimic_lg.scripts.train_velocity import oom_hint
    args=SimpleNamespace(num_envs=2048,replay_capacity=256,batch_size=1024,horizon=16,history=4,
                         terrain="rough",wm_horizon=8)
    hint=oom_hint(RuntimeError("Graph launch error: Warp CUDA error 2: out of memory"),args,free_gib=0.4)
    assert hint["error"]=="cuda_out_of_memory"
    assert hint["suggested"]==dict(num_envs=1024,replay_capacity=128,batch_size=512,horizon=8)
    assert hint["current"]["num_envs"]==2048 and hint["free_gib_on_failure"]==0.4
    small=oom_hint(RuntimeError("out of memory"),SimpleNamespace(num_envs=1,replay_capacity=2,
                    batch_size=1,horizon=1,history=3,terrain="flat",wm_horizon=8))
    assert small["suggested"]["num_envs"]==1 and small["suggested"]["replay_capacity"]==8
    assert small["suggested"]["batch_size"]==1 and small["suggested"]["horizon"]==1

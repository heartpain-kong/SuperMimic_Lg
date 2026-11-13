"""Compare execution paths transactionally on one fixed learner batch.

Tuning never advances saved training weights, optimizer moments, EMA counters,
gradients or RNG. Compilation caches may be retained for the selected path.
"""
import copy
import random
import statistics
import time
import numpy as np
import torch


def tune_rollout(learner, rows, candidates=('batched','compiled'), warmup=1, samples=2):
    if warmup < 1 or samples < 1 or not candidates:
        raise ValueError('tuning needs >=1 warmup, >=1 timed update and candidates')
    if any(mode not in ('reference','batched','compiled') for mode in candidates):
        raise ValueError('invalid runtime tuning candidate')
    if candidates[0] == 'compiled':
        raise ValueError('the first tuning candidate must be an eager baseline')
    start = time.perf_counter()
    device = next(learner.policy.parameters()).device
    models = (learner.wm,learner.policy,learner.critic)
    params = [p for model in models for p in model.parameters()]
    saved = dict(models=[copy.deepcopy(model.state_dict()) for model in models],
        optimizers=[copy.deepcopy(opt.state_dict()) for opt in learner.optimizers],
        learner=copy.deepcopy(learner.state_dict()), gradients=[None if p.grad is None else p.grad.detach().clone() for p in params],
        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        pending=list(learner.timings.pending), runtime=learner.rollout_mode)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def restore():
        for model,state in zip(models,saved['models']):
            model.load_state_dict(state)
        for opt,state in zip(learner.optimizers,saved['optimizers']):
            # load_state_dict can reuse input tensor storage. Each trial needs
            # an independent copy so updates cannot mutate the saved snapshot.
            opt.load_state_dict(copy.deepcopy(state))
        learner.load_state_dict(saved['learner'])
        for p,grad in zip(params,saved['gradients']):
            p.grad = None if grad is None else grad.clone()
        torch.set_rng_state(saved['rng'])
        if saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'],device)
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        learner.timings.pending = list(saved['pending'])
        torch.clear_autocast_cache()

    results, baseline = {}, None
    try:
        for mode in candidates:
            restore()
            learner.rollout_mode = mode
            try:
                learner.timings.reset()
                for _ in range(warmup):
                    learner.update_policy(rows)
                sync()
                durations = []
                for _ in range(samples):
                    t = time.perf_counter()
                    learner.update_policy(rows)
                    sync()
                    durations.append(time.perf_counter()-t)
                if baseline is None and mode == candidates[0]:
                    baseline = [copy.deepcopy(model.state_dict()) for model in models]
                elif baseline is None:
                    raise RuntimeError('eager baseline failed; cannot validate candidate')
                else:
                    for model,reference in zip(models,baseline):
                        torch.testing.assert_close(model.state_dict(),reference,rtol=.02,atol=2e-4)
                results[mode] = dict(status='ok',seconds=durations,median_s=statistics.median(durations))
            except Exception as error:
                results[mode] = dict(status='failed',error=f'{type(error).__name__}: {error}'[:500])
                if mode == 'compiled':
                    learner._compiled_step = None
                if device.type == 'cuda' and 'out of memory' in str(error).lower():
                    torch.cuda.empty_cache()
    finally:
        restore()
        learner.rollout_mode = saved['runtime']
    successful = [mode for mode in candidates if results[mode]['status']=='ok']
    selected = min(successful,key=lambda mode:results[mode]['median_s']) if successful else 'batched'
    # Two samples are a short startup probe. Keep the eager path when a compiled
    # "win" is within 5% noise, rather than switching on a tiny measured gap.
    if selected == 'compiled' and results.get('batched',{}).get('status') == 'ok':
        if results['compiled']['median_s'] > .95*results['batched']['median_s']:
            selected = 'batched'
    if selected != 'compiled':
        learner._compiled_step = None
    return dict(selected=selected,candidates=results,total_s=time.perf_counter()-start,
                warmup=warmup,samples=samples,training_state_restored=True,
                candidate_model_parity=dict(rtol=.02,atol=2e-4),minimum_compiled_gain=.05)

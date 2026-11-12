# XMimic

**Whole-body motion tracking for the Unitree G1, trained 3.7× faster.** XMimic is BlinkMimic's world-model trainer rebuilt
in JAX. On four H100s it reaches BlinkMimic's target in 0.86 minutes, where BlinkMimic needs 3.19 on one; on a single
H100 it is 1.3× faster. Same task, same hyperparameters, same evaluation.

Paper: [XMimic and XSoma: Learning Humanoid Motion Tracking and Locomotion in Under a Minute](https://nico-bohlinger.github.io/papers/2026_xmimic_xsoma.pdf)
(Nico Bohlinger, Jan Peters, 2026), which presents XMimic together with [XSoma](https://github.com/nico-bohlinger/xsoma), the
same recipe for locomotion on rough terrain. Please [cite it](#citation) if you use XMimic. It replaces the first version, which covered
XMimic alone.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/time_to_target_dark.png">
  <img alt="Time to target on H100: BlinkMimic 3.19 min, InstantTrack v1.3 3.31 min, XMimic on 1, 2, 3 and 4 GPUs 2.47, 1.39, 1.09 and 0.86 min" src="figures/time_to_target.png">
</picture>

## What XMimic is

[BlinkMimic](https://github.com/fuyichen0725/blinkmimic) trains one policy that tracks all 40 LAFAN1 motion clips on a
simulated Unitree G1. It follows SuperMimic: learn a world model of the robot, then improve the policy by backpropagating
through rollouts imagined in that model.

XMimic implements the same algorithm with BlinkMimic's hyperparameters (`PRESET[main]`, commit `d264658`). What changes
is the machinery:

- everything runs in JAX: the simulator (MuJoCo Warp through MJX), the replay buffer, the world model, the policy and the critic;
- whole training iterations compile into single XLA programs, so the GPU never waits for Python;
- the iteration can be split across 2 to 4 GPUs that work at the same time.

<p align="center"><img src="figures/xmimic.gif" width="720" alt="XMimic: 3.7x faster training to BlinkMimic's target, with the G1 dancing"></p>
<p align="center"><sub>The G1 driven by an XMimic policy after 54 s of training on 4 H100s (the target checkpoint of a 4-GPU
run of the first release), replayed in MuJoCo on a CPU.</sub></p>

## Results

Time to the same target on H100 NVL GPUs. Every run trains for at least 6 minutes; every 100th checkpoint is evaluated on
all 40 clips with 4 replicas each.

| method | GPUs | runs | it/s | first ≥ 0.82 (min) | held ×3 (min) | speedup | GPU-min to target |
|---|---|---|---|---|---|---|---|
| BlinkMimic | 1 | 5 | 9.1 | 2.25 (1.94–2.51) | 3.19 (2.70–3.91) | 1.00× | 3.2 |
| InstantTrack v1.3 | 1 | 2 | 10.2 | 1.90 (1.81–1.99) | 3.31 (2.34–4.29) | 0.96× | 3.3 |
| XMimic · 1 GPU | 1 | 8 | 12.0 | 1.71 (1.54–1.98) | 2.47 (1.98–3.42) | 1.29× | 2.5 |
| XMimic · 2 GPUs | 2 | 4 | 25.2 | 0.93 (0.84–1.03) | 1.39 (1.16–1.81) | 2.29× | 2.8 |
| XMimic · 3 GPUs | 3 | 4 | 36.4 | 0.71 (0.67–0.72) | 1.09 (0.95–1.41) | 2.91× | 3.3 |
| XMimic · 4 GPUs | 4 | 4 | 38.8 | 0.70 (0.69–0.73) | **0.86 (0.77–0.95)** | **3.70×** | 3.4 |

Times are means over runs (range in brackets), in minutes of training with compilation excluded.

- **The target** is BlinkMimic's: normalized evaluation return ≥ 0.82 with none of the 40 clips failing, held for three
  consecutive checkpoints ("held ×3"). "First ≥ 0.82" is the first checkpoint above 0.82, with failures allowed.
- **On one GPU** XMimic is 1.3× faster than BlinkMimic and InstantTrack, so it also needs the fewest GPU-minutes.
- **On four GPUs** it is 3.7× faster in wall-clock time. It uses a little more GPU time in total than BlinkMimic (3.4 vs
  3.2 GPU-minutes): the speedup buys time, not efficiency.
- **Same final quality.** All methods end at about the same return (0.86 to 0.87 after 6 minutes); XMimic gets there
  sooner.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/learning_curves_dark.png">
  <img alt="Learning curves: evaluation return and failed clips against training time" src="figures/learning_curves.png">
</picture>

<details>
<summary><b>Every layout</b>: how XMimic scales from 1 to 4 GPUs</summary>

<br>

`--layout` chooses how an iteration is spread over GPUs (see [How it gets fast](#how-it-gets-fast)). The 4-GPU layout
(AWLP) is the headline above; the plain 1-GPU trainer is the like-for-like comparison. The runs of FP1, FP and ALP train
3500 iterations, as in the first release. No schedule depends on the number of iterations, so their times to the target
compare directly.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/time_to_target_all_dark.png">
  <img alt="Time to target of every layout" src="figures/time_to_target_all.png">
</picture>

| method | GPUs | runs | it/s | first ≥ 0.82 (min) | held ×3 (min) | speedup | GPU-min to target |
|---|---|---|---|---|---|---|---|
| BlinkMimic | 1 | 5 | 9.1 | 2.25 (1.94–2.51) | 3.19 (2.70–3.91) | 1.00× | 3.2 |
| InstantTrack v1.3 | 1 | 2 | 10.2 | 1.90 (1.81–1.99) | 3.31 (2.34–4.29) | 0.96× | 3.3 |
| XMimic, 1 GPU | 1 | 8 | 12.0 | 1.71 (1.54–1.98) | 2.47 (1.98–3.42) | 1.29× | 2.5 |
| 1 GPU + probe process (FP1) | 1 | 4 | 11.8 | 1.83 (1.74–1.91) | 2.42 (1.93–3.04) | 1.32× | 2.4 |
| 2 GPUs: trainer \| prober (FP) | 2 | 4 | 16.0 | 1.59 (1.58–1.60) | 2.10 (1.89–2.32) | 1.52× | 4.2 |
| 2 GPUs: actor \| learner (AL) | 2 | 4 | 25.2 | 0.93 (0.84–1.03) | 1.39 (1.16–1.81) | 2.29× | 2.8 |
| 3 GPUs: actor \| learner \| prober (ALP) | 3 | 2 | 25.8 | 0.90 (0.87–0.94) | 1.28 (1.00–1.56) | 2.49× | 3.8 |
| 3 GPUs: actor \| world model \| policy (AWL) | 3 | 4 | 36.4 | 0.71 (0.67–0.72) | 1.09 (0.95–1.41) | 2.91× | 3.3 |
| 4 GPUs: actor \| world model \| policy \| prober (AWLP) | 4 | 4 | 38.8 | 0.70 (0.69–0.73) | **0.86 (0.77–0.95)** | **3.70×** | 3.4 |

</details>

<details>
<summary><b>How we measured</b></summary>

<br>

- **Hardware.** One H100 NVL per GPU, the GPUs of a node connected by NVLink. All methods ran on the same cluster.
- **Training budget.** A checkpoint every 100 iterations for every method, as in BlinkMimic, and at least 6 minutes of
  training: BlinkMimic trains its 3500 iterations (6.4 minutes), InstantTrack 4000 (6.6 minutes), and XMimic stops after
  6.2 minutes (`--iterations 100000 --max_minutes 6.2`). FP1, FP and ALP train 3500 iterations.
- **Evaluation.** BlinkMimic's protocol: each clip is played once from its first frame per replica (4 replicas), with no
  randomization and no noise; a clip fails when the training termination triggers. XMimic's checkpoints are scored by its
  JAX evaluator (`python -m xmimic.evaluate`), BlinkMimic's and InstantTrack's by their own `eval_batch.py`. On the same
  checkpoints the two evaluators agree to +0.001 in mean return.
- **Training time.** XMimic compiles before its clock starts. BlinkMimic and InstantTrack compile and capture CUDA graphs
  inside their training loop, so that one-time cost is subtracted from their checkpoint times
  ([`tools/exclude_compile.py`](tools/exclude_compile.py), from their own train logs).
- **Baselines.** BlinkMimic `d264658` (`PRESET[main]`) and InstantTrack v1.3 `9474e74` (its own preset), both unmodified,
  on their PyTorch stack (torch 2.9, mjlab 1.6, MuJoCo Warp 3.11).
- **Data.** Every number above is computed from the evaluation files in [`results/`](results/) by
  [`tools/make_figures.py`](tools/make_figures.py).

</details>

## How it gets fast

An iteration collects about 2.5 simulation steps of 4096 worlds, fits the world model 5 times, and updates the policy by
backpropagating through 32 imagined steps. The work is a long chain of small GPU kernels, so most of the speed comes from
keeping the GPU busy and running parts of the chain side by side.

1. **Whole iterations as compiled programs.** On one GPU, 50 iterations run as one XLA program. Nothing returns to Python
   in between, and checkpoints are written from a background thread.
2. **bf16 where PyTorch uses it.** Matmul outputs are bf16, as in torch autocast, so the backward pass runs on bf16 tensor
   cores too (policy update on an H100: 41.6 → 29.9 ms).
3. **Fewer, larger kernels** in the learner ([`xmimic/fast.py`](xmimic/fast.py)): work that does not depend on the imagined
   state is hoisted out of the 32-step loop, weight gradients are formed in one GEMM per layer, same-shape Muon updates
   are batched, and cuBLAS replaces XLA's Triton GEMMs on Hopper. Each change is tested against the reference on the CPU.
4. **Probe sweeps off the critical path.** BlinkMimic periodically replays the averaged policy on all clips to find
   failures and focus training on them (the paper calls this the failure search; the code's flags are `--eprobe_*`). On
   one GPU a sweep halves the training speed while it runs. XMimic can run the
   sweeps on their own GPU, or in a second process that shares a GPU through CUDA MPS.
5. **An iteration split across GPUs.** An actor collects data, a world-model learner fits the world model, and a policy
   learner runs the backpropagation, all at the same time. Each works on the others' results from the previous iteration.
6. **The host out of the way.** The actor runs in its own Python thread, collection loops get their trip count from the
   host so that launching them never waits for the GPU, the critic fit moves to the world-model GPU, and Python's garbage
   collector is frozen after setup. Together: 20 → 39–41 iterations per second on 4 GPUs.

| `--layout` | GPUs | what each GPU does |
|---|---|---|
| `1GPU` (default) | 1 | everything, probe sweeps inside the loop as in BlinkMimic |
| `FP1` | 1 | the 1-GPU trainer; probe sweeps in a second process on the same GPU (needs CUDA MPS) |
| `FP` | 2 | trainer \| prober |
| `AL` | 2 | actor (collection, world model) \| policy learner; probe process on GPU 1 (needs CUDA MPS) |
| `ALP` | 3 | actor \| policy learner \| prober |
| `AWL` | 3 | actor \| world-model learner and critic \| policy learner; probe process on GPU 1 (needs CUDA MPS) |
| `AWLP` | 4 | actor \| world-model learner and critic \| policy learner \| prober |

On 4 GPUs the actor's 2.5 physics steps (about 24 ms) and the policy update (about 23 ms) are now both on the critical path
of a 25 ms iteration. More speed needs faster physics or collection spread over more GPUs.

<details>
<summary><b>What did not help</b></summary>

<br>

- Overlapping collection and learning on one GPU: XLA stream annotations crash with MuJoCo Warp's kernels or run the
  learner 1.9× slower, and actor and learner as two processes sharing one GPU are no faster than the fused iteration.
- Static collection loop bounds inside the 1-GPU program (3 % slower) or one program per iteration (18 % slower). Only the
  multi-GPU actors use static bounds.
- Other XLA flags: within 1 %, or much slower. Concurrent command-buffer scheduling helps the learners by 1–3 % but slows
  the physics, so it is applied to the learner programs only (`--learner_xla`).
- Two whole trainers on one GPU under CUDA MPS: 1.08× the throughput of one.
- Smaller solver budgets are faster but change the dynamics. Other MJX graph modes, SAP broadphase and fused substeps gave
  nothing.
- Algorithm changes (a higher policy learning rate, larger world-model batches, a lower update-to-data ratio) were within
  seed noise or worse, so BlinkMimic's settings stay.

</details>

## How to run

Linux with NVIDIA GPUs and CUDA 12, Python 3.11. The published numbers are from H100s.
Warp's wheel on PyPI needs an NVIDIA driver of version 580 or newer; with an older driver, install Warp's CUDA 12 build
(`warp_lang-1.19.0+cu12`) from [its releases](https://github.com/NVIDIA/warp/releases) instead.

```bash
pip install -r requirements.txt
bash tools/download_data.sh        # 40 G1 LAFAN1 clips and the G1 meshes, checked against tools/data_sha256.txt
```

Train and evaluate on one GPU (about 5 minutes of training on an H100):

```bash
python train.py --log_dir runs/1gpu
python -m xmimic.evaluate runs/1gpu    # writes runs/1gpu/eval_jax_r4.csv and prints the time to target
```

The 4-GPU layout:

```bash
python train.py --layout AWLP --log_dir runs/awlp
```

`FP1`, `AL` and `AWL` share a GPU between two processes; start CUDA MPS first:

```bash
nvidia-cuda-mps-control -d
python train.py --layout AWL --log_dir runs/awl
echo quit | nvidia-cuda-mps-control
```

The defaults are the published configuration; [`xmimic/config.py`](xmimic/config.py) lists every option. `train.py` sets
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, 90 % GPU memory preallocation and a persistent compilation cache in
`~/.cache/xmimic/jax` unless you set them yourself (`JAX_COMPILATION_CACHE_DIR` moves the cache). The programs of a
configuration compile in parallel threads before the training clock starts, which takes about 2 to 3 minutes the first
time; later runs load them from the cache in about 20 seconds.

Other tools:

```bash
python tools/make_figures.py                     # the figures and tables above, from results/
JAX_PLATFORMS=cpu python tests/test_fast_equiv.py  # fast learner vs reference, on the CPU (a few minutes)
python tools/rollout_cpu.py runs/awlp/ckpt_002000.npz --out rollout.npz   # play a policy on the CPU
python tools/render_gif.py rollout.npz           # the GIF above
```

## Faithfulness and caveats

- **Same algorithm.** In a side-by-side check, XMimic's training statistics (world-model loss, imagined survival,
  simulated return, episode length) were close to BlinkMimic's at every logged iteration, and both reach the same final
  evaluation return.
- **Multi-GPU staleness.** In the actor-learner layouts the actor collects with a policy one iteration old, and the policy
  learner imagines with a world model one iteration old (AWL and AWLP: two). InstantTrack v1.3 uses the same one-iteration
  staleness on one GPU. With the critic on the world-model GPU, the critic is one more iteration old.
- **Constraint budget.** XMimic simulates with 256 constraint rows per world instead of BlinkMimic's 500 (7 % faster per
  physics step). In most runs a training world exceeds 256 a few times (about 6 events in 3500 iterations) and loses the
  extra rows for that step. Evaluation uses 500.
- **Fewer probe sweeps.** In the multi-GPU layouts a sweep runs beside training and spans more iterations than
  BlinkMimic's in-loop sweep (about 30). A sweep that falls due while one runs is skipped, so the 4-GPU layout sweeps
  once every 250 iterations, AL every 255 and AWL every 380, against every 125 in BlinkMimic (and in FP, FP1 and ALP).
  It still reaches the target first.
- **bf16.** The fast learner rounds to bf16 in different places than the reference. Its gradients stay within the
  reference's own bf16 error and the error of torch autocast ([`tests/test_fast_equiv.py`](tests/test_fast_equiv.py)).
- **Noise.** The held ×3 time of a single run can lie up to about a minute from the mean of its method, because it
  depends on when the last hard clips stop failing. Compare means. InstantTrack and ALP have only 2 runs. Throughput
  (it/s) is close to deterministic.
- **Hardware.** All numbers are from H100s. The kernel-level gains target Hopper GPUs and may be smaller on others.

## Repository

```
train.py                 entry point (--layout, all options in xmimic/config.py)
xmimic/                  the trainer: simulator, task, networks, the reference iteration (algorithm.py),
                         the fast learner (fast.py), the layouts (single_gpu.py, multi_gpu.py), the evaluator
assets/g1/               the G1 scene and metadata, exported from BlinkMimic's mjlab scene (meshes are downloaded)
results/                 per-run evaluation files behind every number and figure
figures/                 figures and the GIF
tools/                   data download, figures, compile-time exclusion, CPU rollout, GIF, BlinkMimic interop
tests/                   CPU equivalence tests of the fast learner
```

## Acknowledgements

XMimic stands on other people's work:

- **[BlinkMimic](https://github.com/fuyichen0725/blinkmimic)** by Fuyi Chen: the algorithm, hyperparameters, robot setup
  and evaluation protocol that XMimic reimplements; much of XMimic is a port of its code.
- **[InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack)** by Huy-Quang-Dao, which BlinkMimic builds on and which
  is our second baseline, and **[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)** by Chong Zhang (ETH
  Zurich), the reimplementation both started from.
- **SuperMimic** (Liao et al., arXiv:2610.28480) for the method. XMimic is an independent work, not affiliated with or
  endorsed by the SuperMimic authors.
- **InstantMimic** (Choi, Leem, Won, arXiv:2609.09821) for the idea of a training loop that lives on the GPU.
- [MuJoCo](https://mujoco.org), [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp),
  [NVIDIA Warp](https://github.com/NVIDIA/warp), [JAX](https://github.com/jax-ml/jax) and
  [mjlab](https://github.com/mujocolab/mjlab).
- **Data.** LAFAN1 motions by Ubisoft La Forge (Harvey et al., SIGGRAPH 2020; CC BY-NC-ND 4.0) and the Unitree G1
  description (BSD-3-Clause), via the Hugging Face dataset `lvhaidong/LAFAN1_Retargeting_Dataset`. Neither is
  redistributed here. The GIF and trained checkpoints derive from LAFAN1: non-commercial use only.

Licenses and notices of the code XMimic derives from: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

If you use XMimic in your research, please cite:

```bibtex
@misc{bohlinger2026xmimicxsoma,
  title  = {XMimic and XSoma: Learning Humanoid Motion Tracking and Locomotion in Under a Minute},
  author = {Bohlinger, Nico and Peters, Jan},
  year   = {2026},
  url    = {https://nico-bohlinger.github.io/papers/2026_xmimic_xsoma.pdf}
}
```

XMimic reimplements BlinkMimic and SuperMimic, so please also cite them (see [Acknowledgements](#acknowledgements)).

## License

XMimic is released under the [MIT License](LICENSE). The code it derives from (BlinkMimic, InstantTrack,
supermimic_vibed) is MIT-licensed as well; their notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The
LAFAN1 motions are not part of the repository, and anything derived from them (the GIF, trained checkpoints) is under
CC BY-NC-ND 4.0: non-commercial use only.

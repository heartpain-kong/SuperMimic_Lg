# 0.5.0：对照本地参考实现继续优化

## 参考来源与落地内容

本轮阅读用户提供的 `E:/足式机器人/SuperMimic/代码`，主要采用两种计算组织方式：

1. `xmimic-main/xmimic-main/xmimic/fast.py`：将奖励、soft termination、足端运动学移出逐步递推，按 H×B 批量计算；每帧观测只归一化一次；critic 复用想象阶段特征，固定回归输入只归一化并转换精度一次。
2. `SuperMimic-Reproduced_CuboiLeo/src/supermimic/imagination/compiled.py` 与 `policy_trainer.py`：把固定形状的一个递推步封装成独立可编译函数，优化器更新放在图外。该仓库读取时提交为 `d9be385b3eeb13a638fb33fb200d3bfd9be9839f`。

具体路径、SHA-256 和验证数据见 [本轮记录](validation/2026-11-13_reference_runtime_validation.json)。上游 MIT 许可与说明继续保留在 `docs/THIRD_PARTY_NOTICES.md` 和原版源码快照中。

LgDog 保留自己的奖励、地形、积分和观测规则。网络结构、参数键、检查点 v5/v6 契约不变，不移植 G1 专用目标编码或参考动作接口。

## 三种执行方式

| `--rollout-mode` | 用途 |
|---|---|
| `batched` | **默认**。H 步只递推状态、动作、历史；奖励/失败风险按 H×B 一次计算，复用 WM 物理特征 |
| `reference` | 逐步奖励与终止的对照路径；同样包含 BF16 缓存正确性修复和 critic 输入预处理 |
| `compiled` | **实验选项**。在批量路径上，用 `torch.compile(dynamic=False, fullgraph=True)` 编译固定形状的单步递推 |

工作台“训练效率与显存”可选择执行方式。4060 起步按钮恢复到 `batched`，避免第一次试跑包含编译冷启动。修改表单只影响新进程。

`compiled` 使用 PyTorch Inductor；CUDA 路径需要兼容的编译/Triton 环境。首次前向和反传会编译，冷启动也计入训练墙钟预算。它没有实现原版 JAX 的整迭代 XLA、多 GPU流水线或 deferred decoder weight GEMM。

已用 CPU `aot_eager` 后端验证 FP32/BF16 编译单步的 fullgraph 捕获、输出和反传；**没有在本机验证 CUDA Inductor 编译或加速**。若编译失败或冷启动不适合当前预算，显式选择 `batched`。

## BF16 旧权重缓存修复

旧 runner 在一个 autocast 上下文中执行多个优化器步骤。实验复现：Linear 权重从 1 更新到 0.8 后，同一 autocast 内的下一次前向仍输出 2.0；清除缓存后输出 1.6015625，与当前 BF16 权重相符。

现在每次 `optimizer_step` 后显式清除 autocast 参数缓存，并在无梯度 critic 目标计算和有梯度拟合之间清除缓存。这不取消混合精度，确保每一次更新都使用当前参数。CPU 回归先在旧代码失败、修复后通过；原生 CUDA 回归已提供，本机缺少 CUDA 而跳过。

这是独立的训练正确性修复。它可能影响旧版多次 BF16 更新的数值轨迹，但不能据此断言是用户那条饱和策略的唯一原因。[PyTorch AMP 文档](https://docs.pytorch.org/docs/2.15/amp.html#autocasting)说明了推荐的前向/反传上下文边界。

## 同工作量对照

Windows / PyTorch 2.14.1 CPU，单线程；相同 seed、站立状态批次、batch256、H32、WM 1024×3、actor 512/256/128、critic 256/128。每次 actor 更新仍有 20 次 critic 更新和一次 EMA；每端预热 1 次、正式测量 3 次。

| 对照轮次 | 0.4.0 中位秒数 | 0.5.0 batched 中位秒数 | 比例 |
|---|---:|---:|---:|
| A | 2.4549 | 2.0808 | 1.18× |
| B | 3.0094 | 2.0633 | 1.46× |

基线有系统波动，保守按约 **1.18× CPU 实现收益**描述。该数字不包括减少策略更新次数、不代表 CUDA 性能或收敛速度。

辅助 batch64/H8 profiler：ATen 调用从 0.4.0 的 82575 次降到 39917 次，约减少 51.7%。这是 CPU dispatcher 调用数，不是 CUDA kernel 计数。小批次 CPU 时间未见稳定改善，不能把调用数减少直接等同于所有规模的提速。FP32 完整批次的最终目标/critic loss 等记录一致或仅有微小浮点差异。

正确性测试覆盖 H1/H8/H32、历史 1/4 帧的状态、动作、奖励、终止、生存权重、bootstrap/critic 特征，以及状态、地形、历史、前一动作、指令、相位和 actor 参数的梯度。使用非零 action→state Jacobian 和越界动作检验反传。BF16 对照、固定 critic 输入复用和编译单步也有独立回归。

全套 **104 passed, 3 skipped**；跳过项需要 CUDA。实际 0.4.0 检查点完成一次完整续训，旧 Muon/Adam/critic/EMA 状态正常恢复。工作台通过 Headless Edge 的三种执行方式预览、配置恢复及手机宽度检查。

## 更新与 GPU 实测

```bash
conda activate unitree_rl_mjlab
cd /home/heartpain/SuperMimic
git pull --ff-only origin main
python -m pip install --no-deps -e .
```

更新并重启工作台后，先保持 `batched` 和每轮 BPTT 更新 1 次。已严重动作饱和的旧 actor 建议保留用于排查，以新的起步预设新建训练。

对 GPU 比较两个相同工作量的路径：

```bash
python -m supermimic_lg.scripts.benchmark_learner \
  --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --device cuda:0 --precision bf16 --rollout-mode reference \
  --batch-size 256 --horizon 32 --warmup 3 --updates 10 \
  --output logs/learner_reference_050.json

python -m supermimic_lg.scripts.benchmark_learner \
  --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --device cuda:0 --precision bf16 --rollout-mode batched \
  --batch-size 256 --horizon 32 --warmup 3 --updates 10 \
  --output logs/learner_batched_050.json
```

再用相同参数单独试 `--rollout-mode compiled`；预热会包含首次编译，正式计时只统计预热后的更新。该基准测固定批次的学习器执行时间，正式训练仍需检查物理吞吐和无噪声 EMA 的控制质量。

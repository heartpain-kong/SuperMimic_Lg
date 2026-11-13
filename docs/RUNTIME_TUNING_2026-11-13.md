# 0.6.0：可回退的执行方式实测与完整 4060 参数

完整起步配置见 [XMIMIC_4060_PARAMETERS.md](XMIMIC_4060_PARAMETERS.md)，可执行命令见 [xmimic_4060_flat.sh](../examples/xmimic_4060_flat.sh)。这套配置保留 H32、网络规模和每次 actor 对应的 20 次 critic 更新，显式使用每轮 1 次 actor 更新、actor 基础学习率 1e-4、20 轮预热和 1200 秒试跑预算。它与旧日志每轮 30 次 actor/600 次 critic 的工作量不同；不能把这个调整计作同工作量实现加速。

## 默认批量路径

在 0.5.0 的 H×B 奖励与特征复用基础上，将相位旋转系数、整段指令相位和归一化标准差移出递推循环。标准差每次想象重新读取，不跨训练更新缓存。自定义任务的 advance 保留逐步调用，状态/动作/奖励的时间对齐不变。原版来源仍见 [0.5.0 参考说明](REFERENCE_RUNTIME_2026-11-13.md)。

同一 Windows CPU、PyTorch 2.14.1、单线程、batch256/H32、相同 seed/网络、每次 20 次 critic 更新：0.5.0 中位 1.8806 秒，本轮 1.9150 秒，**没有稳定的额外吞吐提升证据**。辅助 batch64/H8 的 ATen 调用从 39917 降至 39467（约 1.13%）；这不是 CUDA kernel 数，也不代表 GPU 提速。未测 RTX 4060 的 CUDA/Inductor 性能或策略收敛。

## 实验性 auto 模式

CLI 和工作台增加 `--rollout-mode auto`；默认仍为 `batched`，4060 按钮也保持 `batched`。

CUDA 上首次策略更新使用同一批次，分别试 `batched` 和 `compiled`，每种预热 1 次、正式计时 2 次；包含完整 actor 反传、critic 拟合和 XMimic EMA。正式计时前后做 CUDA 同步。随后恢复试跑前的模型、归一化缓冲、优化器动量、EMA、迭代与更新计数、梯度及 Torch/CUDA/Python/NumPy RNG，再执行真正的训练更新。编译缓存可保留给最终选中的路径。

编译路径需通过试跑后模型状态近似检查（rtol=0.02、atol=2e-4），且中位耗时至少快 5% 才选用。编译错误、数值检查失败、编译显存不足或收益不到门槛时使用 `batched`。该短检查用于拒绝明显的数值偏差，不能证明长期训练效果等价。CPU 上直接选 `batched`，不编译。显式 `compiled` 模式仍按原行为运行，不自动回退。

编译与试跑的真实墙钟时间也计入本次时间预算；时间预算在每轮结束检查，不能中途打断编译。20 分钟起步试跑建议继续用 `batched`。`auto` 选择只对当前进程生效，续训时按启动参数重新选择。

启动日志输出 `runtime_tuning` 报告，metrics 包含 `runtime/compiled_selected`、`runtime/reference_selected`、`runtime/tuning_total_s`。试跑更新不计入正式更新计数和阶段耗时；首轮总策略耗时会包含实测选择。

独立比较命令：

```bash
python -m supermimic_lg.scripts.benchmark_learner \
  --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --device cuda:0 --precision bf16 --rollout-mode auto \
  --batch-size 256 --horizon 32 --warmup 1 --updates 10 \
  --output logs/learner_auto_060.json
```

报告保存最终选择、各候选耗时/错误和阶段耗时。独立基准的预热覆盖选择冷启动，正式训练预算仍包含该冷启动。

## 验证

全套 111 passed、3 skipped（CUDA）、3 warnings（兼容旧 Torch API 和 torch.jit）。包含 H1/H8/H32 与 FP32/BF16 输出/梯度对照、站立相位、定制任务、编译 fullgraph、实测过程恢复、异常/中断恢复与错误候选拒绝。CPU aot_eager 捕获不等于验证了 CUDA Inductor。

0.5.0 完整检查点从第 22 轮续训到第 23 轮，Muon/Adam/EMA 与计数恢复正常。Headless Edge 验证 auto 预览、持久化、4060 按钮恢复 batched 和 390px 布局。完整脚本的 151 个参数 token 通过当前 CLI 解析与学习配置校验。原版 XMimic 快照仍按 manifest 检查，不改其源文件。

详细记录见 [validation](validation/2026-11-13_runtime_tuning_validation.json)。

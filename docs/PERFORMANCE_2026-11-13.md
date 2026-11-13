# 0.4.0：MBRL 策略更新性能优化

## 用户日志中的问题

`lg (3)` 的实际命令包含 `--policy-updates 30`。第 606 轮累计时间为 46513 秒；每轮采集约 0.16 秒、WM 更新约 0.43 秒、策略更新约 78.55 秒。actor 每轮增加 30 次，critic 增加 600 次，符合 30 × 5 epochs × 4 minibatches。UTD=0.4 仅控制每轮交替采集 2/3 个向量化物理步，不会限制策略更新次数。

每次策略更新都执行完整 H32 想象、反传和 critic 拟合。把更新次数设为 1 会减少工作量；本页的实现加速对照保持工作量相同，不将这个参数变化计入性能收益。

日志还显示约 99% 的动作饱和，原始动作均值远超执行范围。性能补丁不会自动修复这份策略权重，建议保留原检查点用于诊断，以新的起步配置开始训练。

## 实现变化

- **四腿运动学按深度批量计算。** 读取实际 XML 的父子关系、关节轴、参考角、关节锚点和 body 朝向；12 个逐连杆计算合并为 3 个四腿批次，输出仍按原 tracked-body 顺序排列。
- **四元数向量旋转使用展开公式。** 避免两次 Hamilton 乘法的临时四维向量，保留非单位四元数的原始数学定义及混合 dtype 行为。逆旋转无需构造设备上的共轭常量张量。
- **MuonAdam 的 Adam 分支使用 foreach。** 按设备、dtype 和叶子 step 分组，保留参数缺少梯度时各自的计数；Muon 的更新方程不变。EMA 参数和浮点 buffer 也分组更新，整数 buffer 仍直接复制。
- **增加细分耗时和更新量。** `updates_this_iteration/wm|actor|critic` 明确记录每轮工作量；`time/actor_rollout_s`、`time/actor_backward_s`、`time/critic_targets_s`、`time/critic_fit_s` 和 `time/policy_ema_s` 累计本轮各阶段耗时。CUDA 使用 Event，在已有轮末同步后读取，不在每个阶段强制同步。
- **工作台显示训练量。** 将 BPTT 次数与想象步长分别说明，显示对应的 actor/critic 更新量；高重复次数给出提示，仍尊重显式参数。旧的浏览器表单值不会被静默改写。

网络结构、参数名、奖励、任务契约和检查点版本不变。验证中将优化前 v6 检查点从第 20 轮完整恢复并运行到第 21 轮，Muon/Adam/critic/EMA 状态均保留。浮点运算重排不承诺跨平台逐 bit 等价。

## 相同工作量的 CPU 验证

基线源码为 `8662aff`。Windows / PyTorch 2.14.1 CPU，单线程，相同随机种子和固定 LgDog 站立状态批次。网络为 WM 1024,1024,1024、actor 512,256,128、critic 256,128；每次策略更新包含 20 次 critic 更新和一次 EMA。

| 协议 | 优化前 | 优化后 | 比例 |
|---|---:|---:|---:|
| batch256 / H32，预热 1 次后测 3 次，中位秒数 | 4.2804 | 2.4666 | **1.74×** |
| batch64 / H8，同工作量单次秒数（辅助检查） | 0.7004 | 0.2481 | 2.82× |
| batch64 / H8，额外 profiler 更新的 ATen 调用次数 | 234641 | 82575 | 减少 **64.8%** |

完整规模两端均执行 4 次 actor 和 80 次 critic 更新（包含预热）；该固定批次上的最终记录指标一致。ATen 次数是 CPU dispatcher 侧调用计数，不能当作 CUDA kernel 数量。小规模单次时间仅作辅助，大规模中位时间作为主要对照。

完整测试为 **91 passed, 2 skipped**；两个跳过项均需要 CUDA。覆盖 MuJoCo 外部位姿/速度对照、双精度运动学 gradcheck、非单位四元数输出/梯度、缺失梯度下的 Adam 计数和续训、EMA 整数/浮点 buffer、BF16 物理训练、每轮更新量和工作台。Headless Edge 验证了 30 次 actor / 600 次 critic 的提示、4060 预设和手机宽度。

这些是固定批次的 CPU 实现测试，不代表 4060 加速幅度、完整训练速度或 20 分钟收敛。机器可用 CUDA 时应先运行下面的同工作量基准，再比较完整物理训练的时间和无噪声 EMA 质量。

## 4060 更新和复测

在原环境更新安装并重启工作台：

```bash
conda activate unitree_rl_mjlab
cd /home/heartpain/SuperMimic
git pull --ff-only origin main
python -m pip install --no-deps -e .
```

选择 XMimic-Lg，再点击“填入 4060 起步配置”。此按钮设置 actor 基础学习率 1e-4（当前初期倍率 4，实际 4e-4）、BPTT 更新 1、预热 20、历史 4、回放容量 128、WM/策略批量 512/256、H32、固定前进 0.4 m/s、关闭指令课程以及 1200 秒预算。初始策略和续训路径清空。CLI 算法默认值保持兼容，预设通过显式参数覆盖。

独立测量完整策略更新，不启动 mjlab 大规模物理仿真：

```bash
python -m supermimic_lg.scripts.benchmark_learner \
  --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --device cuda:0 --precision bf16 --batch-size 256 --horizon 32 \
  --warmup 3 --updates 10 --profile --output logs/learner_gpu_040.json
```

基准使用固定状态批次与相同初始网络，输出总时间、细分时间、更新计数和 CPU/CUDA profiler 表。它隔离实现吞吐，不评估走路质量；启用 profiler 时另做一次不计入计时统计的更新。正式训练再检查每轮 `actor=1, critic=20` 是否与所选配置一致。

机器可运行 CUDA 时，原生回归可执行：

```bash
LGDOG_TEST_XML=/path/to/lgdog1.xml python -m pytest tests/test_amp.py tests/test_performance.py -q
```

完整记录见 [性能验证 JSON](validation/2026-11-13_performance_validation.json)。

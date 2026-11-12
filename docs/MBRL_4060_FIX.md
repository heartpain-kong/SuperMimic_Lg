# 4060 速度 MBRL 修复与复测（2026-11-12）

本页记录首轮 revision 4 修复。当前 0.2.0 / modular v5 的接口与迁移见 [重构说明](REFACTOR_2026-11-12.md)。

代码在 `E:/SuperMimic_Lg`；`E:/足式机器人/SuperMimic` 是论文及参考实现资料目录。本次诊断使用用户提供的 `D:/下载/lg` 工作台文本，不包含远程登录信息或可加载的 GPU 检查点。

## 实际情况

- GPU 是 RTX 4060 Laptop、8 GiB；页面快照显存 2.4 GiB、利用率 66%、功耗 40 W。这些是瞬时值，不能推断整段运行的功率限制或 CPU 瓶颈。
- 参数：2048 环境，batch 1024，BPTT horizon 8，history 4，每轮采集 32，WM 更新 20、策略更新 4；命令未覆盖回放容量，按当前默认值 256 估算。
- 日志末尾：14,451 轮、947,060,736 条物理转移、72,627 秒（20.2 小时）。最近一段约 5.04 秒/轮，XY 跟踪误差约 0.30 m/s、yaw 误差约 0.4–0.5 rad/s。
- 这已不是“小数据 CPU 冒烟”。训练数据不足不能解释全部问题。训练奖励包含探索噪声，最终策略质量还须独立无噪声评估。

## 修复内容

1. **输入统计使用原始特征。** 原代码用 `in_norm.update(condition(...))`，而 `condition` 已做归一化。重复更新后均值/方差不再对应物理输入。改为 `raw_inputs` 更新，并用重复输入的统计回归测试检查。
2. **新增实验性 WM 增量损失（默认关闭）。** 原先固定 5 cm 位置、0.5 m/s 速度等容差，会把短期动力学误差压成很小的总损失。通过 `--wm-loss-scale increment` 可从真实转移统计各维增量标准差，并设下限：位置 3 mm、旋转 5 mrad、线速度 0.02 m/s、角速度 0.05 rad/s、关节角 3 mrad、关节速度 0.05 rad/s。启用该实验选项后，新旧 `wm/state_loss` 数值不可直接比较，`wm_online_state_error` 仍保留旧定义便于对照。
3. **越界动作有恢复梯度。** 真实执行和 WM 输入仍限幅到 [-1,1]，额外惩罚 raw actor 输出超出范围的距离平方。避免饱和后只剩零裁剪梯度。测试将输出偏置强制设为 50，确认更新后偏置下降、想象回报保持有限。日志区分 raw 幅度、饱和比例和执行动作标准差。
4. **减少不必要的串行工作。** 每控制步复用上一物理快照；WM 采样不再读取整段 actor 历史；critic 的想象状态特征合并计算；运行归一化不再通过 `float(cuda_count)` 每次强制同步；更新损失汇总后统一取回 CPU。
5. **修正可选深度数据。** 延迟队列每控制步只推进一次，replay 的 `s_t` 配自己的深度图。回归测试检查三步图像序列、深度调用数和状态读取数。
6. **补充诊断。** `time/collect_s`、`time/wm_s`、`time/policy_s`；PyTorch allocated/reserved/peak、全设备占用和回放理论大小。全设备占用包括 Warp、CUDA 和其他进程，不能简单全归入 PyTorch。

检查点保留策略格式 `version=3`，训练标记为 `training_revision=4`，记录所选损失和计算精度；默认 `wm_loss=fixed`。旧 actor 仍可观看/评估；完整旧检查点不能直接 `--resume` 新训练。需要保留旧 actor 时，用 `--init-policy` 且保持原 history / 网络尺寸，重建 WM 和 critic；新配方使用独立训练目录。

## 4060 起步配置

将本次修复包解压到 Ubuntu 的 `SuperMimic` 工程目录，使用原 CUDA Conda 环境安装并重启工作台：

```bash
cd /home/heartpain/SuperMimic
python -m pip install --no-deps -e .
python -m supermimic_lg.workbench --root "$PWD" --port 8089
```

工作台新增“训练效率与显存 → 填入 4060 起步配置”，会设置下列参数，关闭深度并清空续训路径。它是待 GPU 验证的配置，不是 20 分钟收敛承诺：

```bash
python -m supermimic_lg.scripts.train_supermimic \
  --task velocity --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --backend mjlab --device cuda:0 --terrain flat --precision bf16 --wm-loss-scale fixed \
  --num-envs 1024 --iterations 10000 --history 3 \
  --batch-size 512 --horizon 16 --wm-horizon 4 --replay-capacity 64 \
  --collect-steps 8 --wm-updates 20 --policy-updates 1 --warmup-iterations 40 \
  --command-mode fixed --command-vx 0.4 --command-vy 0 --command-yaw 0 --standing-fraction 0 \
  --command-curriculum-iterations 300 --command-curriculum-start 0.5 \
  --action-bound-weight 1 --explore-noise 1 --explore-iterations 100 --action-noise 0.1 \
  --max-training-seconds 1200 --output-dir logs/velocity_4060_fix
```

保留 WM 1024×3 和 actor/critic 512,256,128。BPTT 16 步对应 0.32 秒，先用这一档平衡显存与更新成本；它未覆盖整个 0.6 秒步态周期，是否需要 32 步应由真实评估决定。`max-training-seconds` 从构造 trainer 开始计时，包含初始化；在当前轮完成后保存，可能超过预算一轮及保存耗时。BF16 只对网络算子启用 autocast，物理状态、积分和损失保持 FP32；不支持 BF16 的环境可改 `--precision fp32`。

旧配置回放约 0.758 GiB；新配置约 0.083 GiB（无深度，按实际字段计算）。腾出显存不直接等于提速，也不需要把 8 GiB 填满。先看哪个阶段最慢，再调整该阶段的采集步数、batch 或更新次数；不要只扩大环境数。

## 无探索噪声评估

不需要 HIM 教师即可比较检查点和零动作基线，输出 XY/yaw 误差、前进距离、生存时间、失败率和动作饱和率：

```bash
python -m supermimic_lg.scripts.evaluate_velocity \
  --checkpoint logs/velocity_4060_fix/checkpoint.pt \
  --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --backend mujoco --device cpu --seconds 10 --seeds 0 1 42 \
  --command 0.4 0 0 --output logs/velocity_4060_fix/evaluation.json
```

省略 `--command` 使用统一评估的分段指令。最终应再测试前后、横移、转向及全向范围；单个固定前进结果不能代表全向运控。开启深度的检查点评估目前采用理想深度，不重放训练时的噪声/丢帧/延迟；报告会注明。

真实数据上的输入统计和损失缩放对照可重跑：

```bash
python -m supermimic_lg.scripts.validate_velocity_model \
  --robot-xml /path/to/lgdog1.xml --seeds 0 1 42 \
  --output logs/wm_ablation.json
```

8 个原生 MuJoCo 环境分别采集随机动作；前 4 个拟合，后 4 个验证。三组使用相同数据、初始化、采样窗口和 400 次更新：旧统计更新、仅修统计、统计修复＋增量缩放。分别报告 1/8/16 步开环误差和线速度 RMSE。小规模预测实验不等于策略收敛实验。

三个种子（0、1、42）的均值如下。线速度 RMSE 为三个轴的均方根误差；开环指标是窗口内所有有效步的平均，不是仅最后一步：

| 指标 | 旧统计＋固定损失 | 仅修统计 | 修统计＋增量缩放 |
|---|---:|---:|---:|
| 1 步线速度 RMSE（m/s） | 0.07861 | 0.07969 | 0.04874 |
| 8 步线速度 RMSE（m/s） | 0.11576 | 0.12455 | 0.06784 |
| 16 步线速度 RMSE（m/s） | 0.12652 | 0.14774 | 0.07612 |
| 1 步整体状态误差（统一旧定义） | 0.10220 | 0.11687 | 0.11943 |
| 8 步整体状态误差（统一旧定义） | 0.12044 | 0.13975 | 0.12696 |
| 16 步整体状态误差（统一旧定义） | 0.14564 | 0.17154 | 0.13990 |

16 步速度 RMSE 下降约 **39.8%**，每个种子都下降；整体 16 步误差改善约 3.9%，而整体 1/8 步误差变差。只修正输入统计没有改善预测；本次有效的速度预测改善来自配套的损失缩放。不能把上述结果概括为“所有状态预测更准”或“策略更快收敛”。逐种子数字及协议保存在 [验证数据](validation/2026-11-12_mbrl_model_ablation.json)。

## 策略回归实验与默认选择

额外做了完整闭环对照：旧提交 `e1dbee0` 与上述增量损失实验，均为 8 个 CPU MuJoCo 环境、120 轮、每轮采集 8、batch 64、WM/actor/critic 隐层 64,64、WM 更新 4、策略更新 2、BPTT 8 步、预热 20、episode 100 步、固定 0.4 m/s。随后从独立重置状态用三个种子各评估 10 秒，关闭动作噪声。

| 无噪声结果 | 旧版 | 默认修复（固定损失） | 增量损失实验 |
|---|---:|---:|---:|
| XY 误差（m/s） | 0.4114 | 0.4025 | 0.5209 |
| 平均生存时间（秒） | 10.0 | 10.0 | 1.28 |
| 失败率 | 0% | 0% | 100% |
| 动作饱和比例 | 96.94% | 79.29% | 64.11% |

零动作基线 XY 误差为 0.3997 m/s。三组都没有在这组小预算内学会走路；增量损失实验的生存能力明显退步，因此**不将其设为默认值或 4060 推荐起点**。默认配置保留固定容差损失：验证中保留了 10 秒生存能力，但没有超过站立基线，也没有测得有意义的 CPU 训练提速。最终本地测试为 57 项通过；工作台配置、参数传递、保存和手机宽度布局也已验证。完整数字见 [闭环验证数据](validation/2026-11-12_mbrl_policy_regression.json)。不能用离线预测误差降低代替闭环控制验证，也不能据小 CPU 实验推断近十亿转移的 GPU 收敛结果。

## 验证边界

Windows 本机有 CPU PyTorch 和 MuJoCo，无可用 NVIDIA/CUDA。真实物理训练、梯度、检查点、深度时序和工作台可本机验证；4060 上的 BF16 性能、真实显存峰值及达到目标质量的墙钟时间仍需在原 Ubuntu 环境运行。修复包不包含旧的 20 小时检查点，也不会替换其训练目录。

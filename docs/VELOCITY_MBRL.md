# LgDog 速度指令运控与统一评估

这是基于在线世界模型和短视野 BPTT 的地形运控扩展。它从当前 actor 的物理交互学习动力学，使用速度任务奖励训练策略，不读取参考动作 NPZ，也不加载专家策略。已有 `--task tracking` 仍执行参考动作跟踪，两类检查点不能混用。这不是论文中大规模人形动作库实验的复现结果。

## Ubuntu 训练

在已部署的 LgDog/mjlab 环境中更新本仓库并启动工作台：

```bash
cd /home/heartpain/SuperMimic
git pull
python -m pip install --no-deps -e .
python -m supermimic_lg.workbench --root "$PWD" --port 8089
```

选择 **LgDog · 速度指令运控**。参考动作输入会隐藏。设置仓库路径、GPU 后端、地形和速度上限即可。建议先用 64 环境、批大小 128、预测步长 16、预热 100 轮检查采集/损失/失败，再扩大规模；这些是起步配置，不是已验证的收敛配方。

CLI 等价入口：

```bash
python -m supermimic_lg.scripts.train_supermimic \
  --task velocity \
  --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --backend mjlab --device cuda:0 --terrain flat \
  --num-envs 64 --iterations 20000 --history 3 \
  --batch-size 128 --horizon 16 --warmup-iterations 100 \
  --command-vx 0.6 --command-vy 0.3 --command-yaw 0.6 \
  --output-dir logs/velocity_flat_seed42
```

默认 `--command-mode random`：三项指令是正负对称上限，单位分别为 m/s、m/s、rad/s。每 5 秒重采样，`--standing-fraction 0.1` 表示约 10% 为站立指令。`--command-mode fixed` 则把三项值作为固定指令，可以填写负值；需要一直运动时设置 `--standing-fraction 0`。actor 输出 12 个限幅关节动作，动作缩放 0.25。默认采用原生 HIM 的物理步长 0.005 s、4 个子步，控制周期 0.02 s。

## 深度感知（可选）

默认策略是**盲的**：只用本体感觉、速度指令和步态相位，地形高度图只喂给世界模型和 critic。加 `--depth` 后，LgDog 在躯干前上方（前 0.16 m、上 0.05 m）挂一台朝下 30° 的针孔深度相机，射线按 `--depth-size` 排成图像，`--depth-fov` 为垂直视场，`--depth-range` 为最大感知距离（超出的像素记为该值）。

三类数据的形状变化：

| 数据 | 关闭深度 | 开启深度（feat=32） |
|---|---|---|
| actor 每帧观测 | 47 | 47 + 32 = 79 |
| 世界模型上下文 | 85（地形补丁） | 85 + 32 = 117 |
| 回放中的深度 | 不存 | `[envs, H×W]`，float16 |

深度由一个小型卷积编码器压成 `--depth-feat` 维向量：它同时进入 actor 的帧和世界模型的上下文，编码器参数属于世界模型，**只由世界模型损失训练**（actor 拿到的是它的输出，BPTT 不会反传到编码器）。想象轨迹里不会重新渲染深度：整段 rollout 复用最后一个真实帧的深度特征，因为世界模型没有深度解码头，机器人短时间内脚下的地形也不变。

命令行：

```bash
python -m supermimic_lg.scripts.train_supermimic \
  --task velocity --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --backend mjlab --device cuda:0 --terrain flat \
  --num-envs 256 --iterations 100 --history 3 \
  --depth --depth-size 18x24 --depth-fov 60 --depth-range 3 --depth-feat 32 \
  --output-dir logs/velocity_depth
```

工作台在“LgDog · 速度指令运控”里勾选**启用前向深度相机**即可，分辨率、视场、距离和编码维度都可填。

### 传感器非理想性

真实深度相机有距离相关的噪声、丢帧和延迟。训练时可以打开：

| 选项 | 默认 | 作用 |
|---|---:|---|
| `--depth-noise` | 0 | 二次误差模型，σ = 系数 × z²（米）。0.02 时 1 m 处 σ≈2 cm，3 m 处 σ≈18 cm |
| `--depth-dropout` | 0 | 每帧随机像素无效，读作最大距离 |
| `--depth-latency` | 0 | 深度比本体感觉晚 n 个控制步（50 Hz 下 2 步 = 40 ms） |

这些都记录在检查点 `metadata` 里，**改了就无法续训**；工作台对应“距离噪声系数 / 像素丢失比例 / 深度延迟”。

建议分两段训练：先用理想深度跑通并确认策略能用上深度，再打开 `--depth-noise 0.02 --depth-dropout 0.05 --depth-latency 2` 做鲁棒性微调（用 `--init-policy` 迁移 actor）。直接上噪声会让前几百轮明显更难。


代价与限制：

- 射线数 = `num_envs × height × width`。1024 环境 × 18×24 = 44 万条射线，在 8 GiB 上要相应下调环境数；分辨率翻倍，射线数翻四倍。
- 原生 MuJoCo 后端逐条调用 `mj_ray`，只适合小规模验证；大规模请用 mjlab GPU 后端。
- 训练与想象都假设相机外参固定；噪声、丢帧和延迟已可配置，但**外参标定误差、运动模糊和真实相机的非高斯误差仍未建模**，真机部署前仍需实测校准。
- 深度改变了策略输入维度，旧检查点会因 `metadata` 不一致被拒绝，需重新训练。

## v3：对照 xmimic 修正 actor、world model 与网络规模

速度模式的 `metadata.version` 已升到 **3**，旧检查点（v1/v2）会被拒绝，需要重新训练。这一版按 `xmimic-main` 参考实现（`xmimic/nets.py`、`algorithm.py`）改了三处核心：

**世界模型**

| 之前 | 现在 |
|---|---|
| 直接把原始特征喂进 MLP（关节速度 ~10 rad/s 与离地高度 0.3 m 同量纲） | 输入做**运行均值/方差归一化**（`RunningNorm`，随权重复存） |
| 输出直接当状态增量 | 输出在**归一化残差空间**预测，再用数据统计的标准差反归一化 |
| 纯 Linear+SiLU | 每个隐层 **Linear → LayerNorm → SiLU**（同 xmimic `layer_norm=True`） |

用同一批真实交互数据、同样的 400 步拟合做对照（4→16 环境、平地、随机动作）：

| 指标 | 无归一化 | 有归一化 | 变化 |
|---|---:|---:|---:|
| 单步误差 | 0.04602 | 0.02636 | **−42.7%** |
| 8 步开环误差 | 0.41615 | 0.38294 | −8.0% |

**策略**

- 去掉输出端的 `tanh()`。tanh 在动作边界饱和，梯度恰好在你最需要的地方消失；xmimic 用的是线性头。
- 输出层权重乘 **0.01**、偏置置零，初始动作接近 0（起步稳、不炸），靠训练长出来。
- 输入按帧做**运行归一化**后再展平。

**critic**

- 输入加运行归一化（同 xmimic 的 `critic_b["norm"]`）。
- 目标从 TD(0) 递归改成 **GAE(λ=0.95)**，用可微终止概率当作 done 掩码。
- 每次策略更新后做 **2 轮 × 4 个小批量**回归，而不是一次全量 MSE。

没有照搬的部分：motion encoder / goal 编码（速度模式没有参考动作）、policy EMA、左右镜像增强、多 GPU 拆分。这些在速度任务上不是瓶颈。



### 为什么速度误差下不去

实测过三次，数字都很一致，说明这不是某一处代码坏了，而是这条链路的三层问题叠在一起。

**第一层：数据里根本没有“大步”动作。** 策略头按 xmimic 的做法用 0.01 缩放初始化，实测输出标准差只有 **5e-4**；采集时唯一的扰动是 `--action-noise 0.1`，于是回放里的执行动作标准差 **0.10**、最大幅值 **0.31**。世界模型只在这一小段动作上拟合过，它对“动作加大到 0.8 会发生什么”没有任何依据，策略自然学不会靠动作前进。

已加入探索调度：`--explore-noise`（默认 1.0）在前 `--explore-iterations`（默认 100）轮里线性衰减到 `--action-noise`。日志新增 `action_std` 和 `exploration_noise`，可以确认探索是否真的覆盖了动作空间。实测开启后 `action_std` 从 0.10 升到 **0.70–0.92**。

**第二层：动作幅度上去了，但没有步态。** 开探索后速度误差仍然是 **0.49**，比站着不动的 0.40 还差——策略学会了大幅乱动，没有学会交替迈步。原因是想象视野太短：`--horizon 8` 只有 0.16 s，而步态周期是 0.6 s，回报根本看不到“蹬地—离地—落地”一个完整周期。把 `--horizon` 提到 32（0.64 s，约一个周期）后，策略输出又缩回 0.10、误差仍 0.49。

**第三层：数据量差两三个数量级。** 上面这些 CPU 试验是 8 环境 × 8 步 × 100–150 轮 ≈ **1 万条**转移；你那次 GPU 训练是 2048 × 32 × 170 ≈ **1.1e7**；而运控策略通常要 **1e8–1e9** 条量级才收敛。170 轮不足以判断“学不会”。

另外，奖励结构里**站着是个很强的局部最优**：站立能拿到约 2.2/步（正立 0.5 + 高度 0.5 + 部分速度项），完全追速只多约 0.9/步，而一旦摔倒这些全丢。在模型不准的早期，策略自然选择“站着不冒险”。

因此下一步的优先级建议是：

1. 先把数据量补上：GPU 上 2048–4096 环境、至少 2000–5000 轮，观察 `velocity_error_xy_mps` 是否单调下降，而不是 170 轮就下结论。
2. `--horizon 32`（约一个步态周期）、`--policy-updates 1`、`--batch-size 256`，避免 8 步视野导致的短视。
3. 加一个**前进进度奖励**（沿指令方向的位移），给“开始动”一个持续梯度，而不是只靠瞬时速度匹配。
4. 指令做课程：先 0.2 m/s 学会走，再放大到 0.6–1.0。

### 网络规模

之前的默认值是 `128×128×128` 的世界模型和 `128` 宽的两层 actor/critic，比社区实现小一到两个数量级（xmimic 的 WM 是 `1024,1024,1024`，HIM 的 actor 首层就是 512）。现在按参考实现把默认值放大：

| 组件 | 之前 | 现在 | 参数量（旧 → 新） |
|---|---|---|---:|
| world model | 128×3 | **1024×3** | 74,153 → **2,427,945** |
| actor | 128×128 | **512,256,128** | 36,236 → **238,476** |
| critic | 128×128 | **512,256,128** | 33,281 → **230,913** |
| 合计 | — | — | 0.5 MB → **11.1 MB**（fp32 权重） |

容量在这台机器上从来不是瓶颈：三个网络加起来只有 11 MB，真正吃显存的是回放缓存和 mjlab 仿真。所以“模型太小”确实是一个真问题，而不是显存取舍。

改法：

```bash
--wm-hidden 1024,1024,1024      # 世界模型隐藏层，逗号分隔
--actor-hidden 512,256,128
--critic-hidden 512,256,128
```

工作台「网络结构（高级）」里有对应三个输入框。更换隐藏层会让旧检查点失效（`metadata` 里记录了这三个列表）。



## 显存与规模

回放缓存按 `replay_capacity × num_envs × (状态 42 + 下一状态 42 + 地形上下文 85 + 历史 4×47 + 动作 12 + 上一动作 12 + 指令 3 + 相位 2 + 2)` 分配，全部常驻显存，因此它随环境数线性增长；BPTT 峰值主要由 `batch_size × horizon` 决定。8 GiB 显卡（例如 RTX 4060 Laptop）上的参考起点：

| 显卡显存 | num-envs | replay-capacity | batch-size | horizon | 说明 |
|---|---:|---:|---:|---:|---|
| 8 GiB | 256 | 64 | 128 | 8 | 先跑通流程，约占 1 GiB 以内 |
| 8 GiB | 1024 | 64 | 256 | 8 | 稳定后扩一档 |
| 8 GiB | 2048 | 32 | 128 | 4 | 上限尝试，仍可能 OOM |
| ≥24 GiB | 2048 | 256 | 1024 | 16 | 工作台默认大配置的量级 |

2048 环境 + 256 回放容量在 8 GiB 上仅回放就约占 0.8 GiB，再加上 mjlab/Warp 的 CUDA graph 与传感器缓冲，通常在开始采集的第 2 轮触发 `Warp CUDA error 2: out of memory`。

训练启动时会打印 `device_memory` 的可用/总显存。若仍发生显存不足，进程会输出 `cuda_out_of_memory` 的 JSON，其中 `suggested` 给出减半后的环境数、回放容量、批大小和预测步长，并尽量保存检查点，然后以退出码 3 结束，便于工作台把这次运行标为失败而不是成功。


## 调整目标和奖励

工作台的“平地前进 / 全向指令 / 地形前进”按钮只填写任务参数。“平地前进”设置固定前进 0.4 m/s、横移和转向为零、站立比例为零；“地形前进”在此基础上选择 rough。它们不是已验证收敛的配方。手动选择“随机范围”时，0.4 表示 -0.4 到 0.4；选择“固定速度”时，0.4 表示固定向前 0.4 m/s。

展开“奖励与目标高度”即可修改 11 项权重、目标机身离地高度、固定抬脚高度和速度奖励宽度。命令行示例（追加到上面的速度训练命令）：

```bash
--command-mode fixed --command-vx 0.4 --command-vy 0 --command-yaw 0 \
--standing-fraction 0 --target-height 0.30 --swing-height 0.06 \
--reward-velocity 2 --reward-yaw 1 --reward-upright 0.5 --reward-height 0.5 \
--reward-gait 0.3 --reward-lift 0.3 --reward-action-rate -0.05 \
--reward-slip -0.05 --reward-energy -0.002 --reward-joint-limit -2 \
--reward-trunk-contact -2 --velocity-tolerance 0.5 --yaw-tolerance 0.5
```

奖励项用非负数，惩罚项用非正数，0 可关闭一项。速度奖励为 `exp(-error² / tolerance²)`，宽度越小越严格；它是训练奖励参数，不会改变评估通过门槛。机身目标高度相对根下地面计算；抬脚目标相对脚下地面计算，尚不按前方台阶高度自动调整。物理失败高度范围、倾斜和机身接触判断也不会随奖励权重改变。

实际仿真和 WM 推演调用同一个奖励函数并使用相同配置。日志会输出 `reward/velocity` 等各项加权贡献，工作台日志中可展开查看；这些贡献之和等于本轮真实平均奖励。WM 的监督目标仍是真实物理转移与接触，不是奖励大小。

配置保存进 config.json 和 checkpoint，并参与共享任务哈希。改变奖励/目标后请创建新训练记录；需要沿用旧速度策略时使用 `--init-policy`，不要使用旧任务的 `--resume`。本次增加配置契约后，旧版本检查点也应采用 `--init-policy` 迁移。任务不会在已运行的训练中热更新。

`train_him_velocity` 接受相同的上述参数。公平比较时，把这些参数同时传给两种算法，并核对 `shared_task_contract_sha256`；不能只修改一边的奖励。

速度模式仍使用世界模型中的可微 BPTT 更新 actor，没有 PPO 概率比裁剪目标。它不包含指定动作的模仿目标，属于 WM+BPTT 运控扩展；需要模仿指定动作时选择 SuperMimic 动作跟踪并提供 NPZ。HIM/PPO 是独立对照入口。

平面速度误差在仅绕 Z 轴旋转的 heading 坐标系中计算，偏航速度使用世界 Z 轴角速度。两种共享任务入口使用同一公式；历史 HIM 策略原先的训练定义可能不同，只能作为诊断基线。

`--terrain rough` 使用 LgDog 原有 mjlab 地形生成器，包含平地、上下台阶、上下坡、随机起伏和波浪。也可选 `stairs_up / stairs_down / slope_up / slope_down / rough_height / wave` 单项。金字塔地形从中央平台向外走：上台阶/上坡选择中央较低的 inverted 变体，下台阶/下坡选择中央较高的变体。当前参数来自 LgDog 仓库的 `terrain_cfg.py`。

仅在实际按指令前进、达到距离/跟踪门槛时提高地形等级，跌倒或明显停滞会降级。站在出生点不会被计为地形通过。

CPU 验证同样不需要 NPZ：

```bash
python -m supermimic_lg.scripts.train_supermimic \
  --task velocity --mjlab-root /path/to/unitree_rl_mjlab_lgdog \
  --backend mujoco --device cpu --terrain stairs_up \
  --num-envs 2 --iterations 20 --batch-size 8 --wm-hidden 32,32 --actor-hidden 32,32 \
  --collect-steps 8 --wm-updates 2 --policy-updates 1 \
  --horizon 4 --warmup-iterations 5 --output-dir logs/velocity_cpu
```

CPU 粗糙地形也使用 mjlab 原生生成器，需要已安装 mjlab 与 LgDog 源码，但物理推进使用原生 MuJoCo；GPU 则使用 Warp。CPU 结果不能验证 CUDA Graph、显存容量或 GPU 吞吐。

## 世界模型看到了什么

- **actor**：每帧 47 维，顺序与盲 HIM 导出契约一致：机身角速度、重力方向、速度指令、相位、相对关节位置、关节速度、上一实际执行动作。没有额外地形扫描或接触标签。历史长度可选，默认 3。
- **WM state**：37 维物理状态，加四足接触与机身接触，共 42 维。物理残差预测和接触概率一起从真实执行转移拟合。
- **WM context**：9×9 地面高度图及其世界坐标锚点/朝向，共 85 维。高度图来自实际地形射线，不使用伪造高度；射线漏检会明确报错。当前摩擦、质量和 PD 固定为名义值，尚未加入域随机化。
- **想象**：每次预测都按预测根位置/朝向重新查询同一静态局部图。双线性查询对预测位置可微。WM 查询中央约 1m 的邻域，给预测位移保留余量；根移出地图锚点 0.5m 时停止推演，此前逐渐降低生存概率。实际仿真每步重新采集完整 2m 图。
- **离地高度与终止**：根高度减根下方地面高度，而非固定世界 Z。离地高度不在 `[0.14,0.65]m`、机身倾斜超过约 60° 或机身接触地面会判为物理失败。回合超时仅是 replay 分段，不是物理失败标签。

奖励包括平面/偏航速度跟踪、姿态、相对离地高度、对角步态接触/抬脚、滑移、能耗、动作变化和关节限位。脚的位置/速度由实际 XML 的可微 FK 计算，再查询脚下地形。WM 的预测损失、接触损失和 actor 的 BPTT 梯度分别记录。

局部图的分辨率为 0.25m，会平滑台阶边缘；固定名义上下文和有限局部图仍限制泛化能力，不能将接口通过当作楼梯策略已经训练成功。

## 保存、续训与平地迁移

`checkpoint.pt` 包含 actor、critic、WM、三个优化器、累计采集量、计时和硬件记录。`--resume` 要求相同机器人、策略结构、地形、速度范围、seed 和任务代码契约；回放、物理状态与地形课程会重建，不保证逐步复现。新任务需要使用新的输出目录，避免混入旧日志。

从已训练好的平地速度策略开始地形训练时，使用：

```bash
--terrain rough --init-policy logs/velocity_flat_seed42/checkpoint.pt
```

它只迁移 actor，重新训练地形 WM 与 critic，并重新计数；不能同时填写 `--resume`。工作台有对应“初始速度策略”字段。速度策略可在“策略观看”中直接加载，无需 motion；观看使用 CPU MuJoCo 的实际选定地形和固定前进指令 0.6m/s。

## 与已有 HIM/PPO 共用评估

工作台选择 **统一评估 · SuperMimic / HIM-PPO**，填写速度模式检查点及现有 `policy.onnx`。两者共用同一个已编译物理模型、PD、步长、初始扰动、命令序列、种子、终止与通过标准。输入历史长度必须相同，例如与 V28 比较时应训练 `--history 6`；不能将 3 帧与 6 帧当成严格消融。

```bash
python -m pip install onnxruntime
python -m supermimic_lg.scripts.benchmark_velocity \
  --checkpoint logs/velocity_flat_seed42/checkpoint.pt \
  --him-onnx /path/to/HIM/policy.onnx \
  --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --backend mjlab --device cuda:0 \
  --terrain stairs_up --level 1 --seeds 0 1 42 --seconds 10 \
  --output logs/comparison/stairs_up.json
```

对 flat、stairs_up、stairs_down、slope_up、slope_down、rough_height、wave 分别运行。拒绝用一个混合地形单环境代表所有地形。

报告包含逐回合速度误差、偏航误差、生存时长、失败、沿指令实际进度、通过率、硬件与协议/模型哈希。通过要求完整生存、实际前进至少 2m 且达到命令距离的 70%，同时平均 XY 速度误差小于 0.8m/s。此定义是公开的任务完成门槛；应同时查看原始误差和逐回合结果，不能只看通过率。

对于严格的训练时间比较，使用新增的 `train_him_velocity`，而不是原先奖励、随机化和重置规则不同的 Rough-HIM 任务。它复用现有仓库的 HIMActor、HIMPPO 和 PPOHIMRunner，采用与 MBRL 相同的速度范围、地形、无随机化物理、重置、奖励、相对高度终止和课程。两者的 actor 都只接收 47 维本体帧；critic 都接收 129 维特权输入。HIM 的估计器、观测归一化、Gaussian 探索及 PPO 更新仍是该算法自身的一部分，完整配置会保存，不冒充神经网络结构完全相同。

两种入口会保存算法无关的 `shared_task_contract_sha256`，涵盖机器人/地形/奖励代码、步长、指令、重置、噪声和输入历史。计时记录可核验两次训练的任务契约及机器/设备是否一致。可先用 `train_him_velocity --check-config` 在 CPU 上检查配置；实际 HIM/PPO 训练要求 CUDA。

下面在同一台 Ubuntu GPU 上依次运行，明确使用同一设备（Bash 数组）：

```bash
COMMON=(--mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog
        --terrain rough --device cuda:0 --num-envs 64 --history 3
        --collect-steps 16 --episode-steps 500
        --torch-threads 4
        --command-vx 0.6 --command-vy 0.3 --command-yaw 0.6
        --seed 42 --iterations 20000)

python -m supermimic_lg.scripts.time_training \
  --algorithm supermimic --output logs/timing/supermimic.json \
  --task-config logs/controlled/supermimic/config.json -- \
  python -m supermimic_lg.scripts.train_supermimic --task velocity --backend mjlab "${COMMON[@]}" \
  --batch-size 128 --horizon 16 --warmup-iterations 100 \
  --output-dir logs/controlled/supermimic

python -m supermimic_lg.scripts.time_training \
  --algorithm him_ppo --output logs/timing/him.json \
  --task-config logs/controlled/him/config.json -- \
  python -m supermimic_lg.scripts.train_him_velocity "${COMMON[@]}" \
  --output-dir logs/controlled/him
```

计时包含子进程初始化、编译及命令内执行的评估；实际配置在子进程结束后读取并保存哈希。工作台评估的高级字段或 CLI 的 `--supermimic-training-manifest / --him-training-manifest` 可把两份耗时记录附入对照报告。首次编译与已有缓存会影响墙钟时间，应给两边相同的缓存条件并保存运行顺序；不要并发争用同一 GPU。需要按多个训练种子重复此过程。

**历史策略的共用评估不代表其训练条件已经相同。** 现有 V28 等 HIM 导出不能证明它原先的硬件、奖励、随机化和训练预算。严格比较应使用上述共享任务入口，在同一机器重新训练，并按相同检查点频率评估达到共同质量门槛的时间。MBRL 的 WM/actor 更新次数、PPO 的 epoch 和 HIM 估计器更新属于各自方法，应记录而不是把同一“迭代数”当成相同计算预算。工具对未知值保留 null，不从最终策略或录制视频推断加速倍数。

## 本次验证边界

Windows CPU 上已验证无 NPZ 的真实物理采集、三个种子的更新/重置/续训、台阶物理采集、地面高度插值梯度、升高地面下的高度/终止不变性和观测隔离。共用评估器也对现有 V28 HIM 和一个仅 3 轮的速度模式集成检查点运行了平地/上台阶的三种子诊断。该新检查点未学会行走，不作为性能或训练加速结果。Ubuntu GPU 端的完整收敛训练与同硬件时间对照仍需实际运行。

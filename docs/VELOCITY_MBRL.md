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

三项指令上限为正负对称范围，单位分别为 m/s、m/s、rad/s。每 5 秒重采样，10% 为站立指令。actor 输出 12 个限幅关节动作，动作缩放 0.25。默认采用原生 HIM 的物理步长 0.005 s、4 个子步，控制周期 0.02 s。

平面速度误差在仅绕 Z 轴旋转的 heading 坐标系中计算，偏航速度使用世界 Z 轴角速度。两种共享任务入口使用同一公式；历史 HIM 策略原先的训练定义可能不同，只能作为诊断基线。

`--terrain rough` 使用 LgDog 原有 mjlab 地形生成器，包含平地、上下台阶、上下坡、随机起伏和波浪。也可选 `stairs_up / stairs_down / slope_up / slope_down / rough_height / wave` 单项。金字塔地形从中央平台向外走：上台阶/上坡选择中央较低的 inverted 变体，下台阶/下坡选择中央较高的变体。当前参数来自 LgDog 仓库的 `terrain_cfg.py`。

仅在实际按指令前进、达到距离/跟踪门槛时提高地形等级，跌倒或明显停滞会降级。站在出生点不会被计为地形通过。

CPU 验证同样不需要 NPZ：

```bash
python -m supermimic_lg.scripts.train_supermimic \
  --task velocity --mjlab-root /path/to/unitree_rl_mjlab_lgdog \
  --backend mujoco --device cpu --terrain stairs_up \
  --num-envs 2 --iterations 20 --batch-size 8 --network-size 32 \
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
        --network-size 128 --collect-steps 16 --episode-steps 500
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

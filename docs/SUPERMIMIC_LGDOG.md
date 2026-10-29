# LgDog 在线 SuperMimic 训练

`supermimic_lg.scripts.train_supermimic` 直接用 LgDog 的 12 关节物理模型执行 SuperMimic 循环：

1. 当前策略在真实仿真中执行动作，记录实际执行的带噪动作及状态转移；
2. 在线回放缓冲采样短窗口，拟合可微动力学和终止头；
3. 从真实状态及其观测历史开始，在冻结的世界模型中做短视野想象；
4. 使用 LgDog 可微 FK 计算 body 跟踪奖励，通过 BPTT 更新策略，并拟合 critic；
5. 新策略返回仿真继续采集，周期保存所有模型和优化器。

该入口使用独立 runner，没有调用 PPO。工作台第一项 **LgDog · SuperMimic 在线训练** 对应此入口；`LgDog1 动作跟踪 · PPO` 是另一个基线入口。

## 数据与机器人

需要 [Ubuntu 部署文档](UBUNTU_22_04.md) 中的 LgDog 仓库和与 XML 匹配的 motion NPZ。参考动作应是 50 fps、12 个关节、13 个 body，顺序按 `config.py` 的 LgDog1 定义，body 数组采用世界坐标中的 link 原点和 wxyz 四元数。初始化会对若干帧做 FK 一致性检查，关节/body 顺序错误或动作来自其他机器人时会报错。

无需预先录制 replay。训练器直接使用仿真的真实转移；发生摔倒、动作片段结束或回合超时后，先记录重置前的物理状态，再重置到参考动作随机帧。跨回合的窗口会掩码，不会把新出生状态当成世界模型预测目标。世界模型终止头只学习物理失败，动作片段结束和回合时限只用于划分回放边界。

状态为 37 维，动作 12 维，默认姿态 `(0, 0, 0.2)`，动作缩放 0.25，控制频率 50 Hz。策略输入含关节位置/速度、机身角速度、重力方向、上一动作及 当前帧及前两帧观测（每帧 42 维，结构为 `[N,3,42]`，网络内部展平为 126 维）；目标由未来参考帧构成。FK 从实际 XML 编译后的树、关节轴和锚点构建，位姿与 link 原点速度均可微。

## Ubuntu / mjlab CUDA

先完成 mjlab 锁定环境安装，激活同一个虚拟环境并更新本仓库：

```bash
source "$HOME/work/lgdog-venv/bin/activate"
cd "$HOME/work/SuperMimic_Lg"
git pull
python -m pip install --no-deps -e .
python -m supermimic_lg.scripts.train_supermimic \
  --mjlab-root "$HOME/work/unitree_rl_mjlab_lgdog" \
  --motion-file /data/lgdog_motion.npz \
  --backend mjlab --device cuda:0 \
  --num-envs 64 --iterations 100 \
  --collect-steps 16 --wm-updates 10 --policy-updates 2 \
  --warmup-iterations 5 --batch-size 128 --horizon 8 \
  --seed 42 --output-dir logs/supermimic_lgdog_seed42
```

mjlab 后端直接构建 LgDog 的 Scene/Simulation、地形和 PD 执行器，不实例化 tracking PPO runner。先从 64 个环境检查真实采集和更新，再根据显存逐步增大环境数。`--iterations` 是总训练迭代目标；一次迭代包含采集、多次世界模型更新和多次策略更新。真实环境转移数等于采集步数乘环境数再乘迭代次数。

### mjlab 风格楼梯、坡道和起伏地形

训练命令添加 `--terrain rough` 可启用 LgDog 仓库 `src/tasks/velocity/config/lgdog1/terrain_cfg.py` 中的地形生成器：平地、上/下台阶、上/下坡、随机高度起伏和波浪。初始分配从低难度行开始；升级需至少 50 个控制步、参考平面位移达到 2 米、实际沿参考方向前进至少 2 米且达到参考位移的 70%；物理失败或明显停滞则降低一档，短片段保持等级。可用工作台里的“训练地形”选择相同配置。

该功能需要 `--backend mjlab --device cuda:0`，原生 MuJoCo CPU 后端仍仅有平地。SuperMimic 仍在跟踪 motion NPZ 提供的未来根/body 轨迹；静止 `standing.npz` 不会教会机器人行走。当前策略没有 terrain height scan，属于仅凭本体历史的盲地形训练。原有策略观看器也使用平地 MuJoCo，不能把粗糙地形检查点当作地形通过验证。训练粗糙地形请提供真实行走参考动作，并在 mjlab 上检查每类地形的通过率和跌倒率。

## 原生 MuJoCo CPU

CPU 后端也使用真实 LgDog 物理，不使用合成动力学替代。它逐个推进环境，适合少量环境验证；神经网络可选择 CPU 或 CUDA。原生后端从同一 XML 去掉视觉网格，保留 inertial、joint、collision，添加平地及 position actuators：hip/thigh `kp=20,kd=1,10Nm`，calf `kp=40,kd=2,20Nm`，armature 分别为 `.01/.02`，与当前 LgDog mjlab 配置对应。

使用 Python 3.10–3.13 和安装了 MuJoCo 的环境：

```bash
python -m supermimic_lg.scripts.train_supermimic \
  --mjlab-root /path/to/unitree_rl_mjlab_lgdog \
  --motion-file /data/lgdog_motion.npz \
  --backend mujoco --device cpu --num-envs 2 \
  --iterations 20 --batch-size 8 --network-size 32 \
  --collect-steps 8 --wm-updates 5 --policy-updates 1 \
  --warmup-iterations 2 --horizon 4 \
  --seed 42 --output-dir logs/supermimic_cpu_seed42
```

仅使用原生后端时可用 `--robot-xml /path/to/lgdog1.xml` 直接指定 XML，省去对 LgDog Python 工程的依赖。MuJoCo 3.5.0 不提供当前 Windows Python 3.14 环境所需的预编译 wheel；请使用上述受支持 Python 环境。

## 检查点与继续训练

每个输出目录包含 `config.json`、`metrics.jsonl` 和 `checkpoint.pt`。检查点默认每 10 次迭代及退出时保存，包含 `wm/policy/critic`、三个优化器、迭代数、更新次数、策略结构、motion/XML 哈希和 CPU Torch RNG 状态。写入先落到临时文件，再替换旧文件。

继续训练时，在原命令上加：

```bash
--resume logs/supermimic_cpu_seed42/checkpoint.pt --iterations 100
```

需要使用相同 motion/XML、`--network-size`、`--history` 和 `--goal-horizon`。当前默认 `--history 3`，表示当前 1 帧加过去 2 帧；旧 10 帧检查点需显式传入 `--history 10`，或在工作台将观测总帧数改为 10。这会恢复模型、优化器和训练计数；物理状态及 replay 会重新采集，先重新预热。它不保证位级重现原有物理轨迹。Linux 的 Ctrl+C/SIGTERM 会尝试保存当前模型；强制结束只能使用最近一次已完成的检查点。

日志分别输出真实仿真奖励、物理失败数、完整回合数/长度、世界模型状态误差与终止损失、想象奖励、生存概率、策略梯度，以及世界模型对动作的梯度。想象奖励不能替代真实环境评估；短站立参考动作的检查通过也不能证明学会了行走。

## 当前验证和范围

原生 MuJoCo 后端已用实际 LgDog XML 检查 FK 位姿、link 原点速度与梯度、三个随机种子的在线更新、检查点重新加载和继续更新。mjlab CUDA 接口已按本地 1.2.0 源码接入，但本机缺少可运行的 CUDA 训练环境，尚未完成 Ubuntu/GPU 实机验证。

2026-10-28 的本机 CPU 集成验证采用站立参考、2 个真实 MuJoCo 环境、小网络，结果如下；这些是训练循环验证，不是学习到步态或优于 PPO 的证据。

| 随机种子 | 迭代 | 实际物理转移 | 世界模型更新 | BPTT 策略更新 | 最后一轮策略梯度范数 |
|---|---:|---:|---:|---:|---:|
| 0 | 20 | 320 | 100 | 19 | 0.003379 |
| 1 | 20 | 320 | 100 | 19 | 0.001298 |
| 42 | 20 | 320 | 100 | 19 | 0.001189 |

当前是单参考片段、固定名义上下文（摩擦 .6、编码器偏置和 COM 偏移为 0）的在线实现。多片段训练、terrain height scan、完整域随机化、论文规模的吞吐和真机性能仍需后续实现/验证。

训练日志的 `transitions` 记录累计真实采集量，`replay_transitions` 记录当前回放容量。新检查点会保存累计量；旧检查点缺少该计数时，从恢复后的采集重新计数。粗糙地形日志另外报告 `terrain/level_mean`、`terrain/promotions` 和 `terrain/demotions`。静止参考动作会在粗糙地形训练启动前报错，避免将站立训练当作地形行走。

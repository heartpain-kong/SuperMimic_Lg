# XMimic 接入：LgDog 适配与原版 JAX 入口

版本 0.3.0 同时加入两条独立入口。工作台算法选择中的 **XMimic-Lg · PyTorch** 使用当前 LgDog 物理任务；包内另有原版 **JAX/XLA · G1** 运行源码和启动器。两者的任务、数据、依赖和检查点分别标识。

## LgDog：直接使用现有 Conda 环境

不需要安装 JAX，也不改变当前 mjlab / CUDA 依赖。在训练工程目录更新安装并重启工作台：

```bash
python -m pip install --no-deps -e .
python -m supermimic_lg.workbench --root "$PWD" --port 8089
```

选择 **LgDog · 速度指令运控 → 训练算法 → XMimic-Lg · PyTorch（Muon＋EMA）**。切换时填入 XMimic-Lg 配置，随后仍可修改批量、网络、奖励等参数；工作台会明确生成 `--algorithm xmimic`。观看和评估默认加载导出的 EMA 策略。

CLI 等价入口：

```bash
python -m supermimic_lg.scripts.train_xmimic \
  --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --backend mjlab --device cuda:0 --precision bf16 --terrain flat \
  --num-envs 1024 --iterations 10000 --history 3 \
  --wm-batch-size 512 --policy-batch-size 256 \
  --command-mode fixed --command-vx 0.4 --command-vy 0 --command-yaw 0 --standing-fraction 0 \
  --max-training-seconds 1200 --output-dir logs/xmimic_lg_4060
```

或使用公共入口 `python -m supermimic_lg.cli --algorithm xmimic --task velocity ...`。现有 `--algorithm supermimic` 和旧模块入口仍保留。上例是用于 GPU 复测的起点，未证明能在 20 分钟内收敛，也没有实测 4060 显存峰值。

### 适配内容

| 项目 | XMimic-Lg 实现 |
|---|---|
| 模型/策略优化器 | 中间 Linear 权重用 Muon，首末 Linear 与向量参数用 Adam；critic 用 Adam |
| Muon | momentum=0.95，5 次 Newton–Schulz，系数 3.4445/-4.775/2.0315，RMS 比例 0.2×sqrt(max(shape)) |
| 策略 EMA | 默认 0.995；第一次更新直接复制，此后平均参数和归一化 buffer；保存并恢复更新次数 |
| 结构化动力学 | 四个物理子步的 0.625 加权速度漂移；适配 LgDog 的 world 线速度 / body 角速度 / heading 残差布局 |
| WM 输出归一化 | eps=1e-4 |
| 调度 | UTD=0.4，每轮采集交替 2/3 步；首次/恢复后先采集 64 步重建 replay |
| 默认想象/拟合 | horizon=32，WM window=8，WM 更新 5，策略更新 1，critic epochs=5 / minibatches=4 |
| 学习率 | 基础 4e-4；actor 初期倍率 4；1400 轮开始用 300 轮降到倍率 1；250000 轮余弦衰减到底值 3% |
| 镜像增强 | 可设置 `--wm-mirror-fraction 0.5`；LgDog 默认 0，见下文 |

CUDA 上 Muon 正交化使用 BF16，CPU 检查使用 FP32。移植的是更新方程与训练组件，不宣称逐 bit 等价于 JAX 的 BF16 累加或 XLA 执行。

本地 LgDog XML 左右前髋原点 X 差 2.35 mm；12 组测试姿态的足端镜像最大位置误差约 0.105 mm，速度误差约 0.036 mm/s。镜像映射通过反射/反射还原和地形采样测试，但物理模型不能视为完全对称，因此**没有默认照搬上游 50% 镜像**。

### 与原版的边界

LgDog 继续使用自己的 12 关节、47 维本体观测、动作限幅、PD、地形/接触上下文和速度奖励。actor 默认 512,256,128，critic 256,128，history 默认 3；模型/策略 batch 为 512/256，而原版使用 G1 与动作目标编码器，默认两个 batch 都是 2048。

LgDog 版保留当前探索调度、奖励范围 ±10、critic 目标范围 ±200，以及默认固定容差 WM 损失。需要实验时可显式设置 `--explore-iterations 0 --reward-limit 0 --value-target-limit 0 --wm-loss-scale increment`；之前的增量缩放曾出现小预算闭环回归，不因此自动开启。

没有在此 PyTorch 适配中加入原版整迭代 XLA 编译、JAX fused learner、多 GPU 流水线、动作库评估探针和基于探针的失败重置。LgDog 学习率使用迭代触发，未移植 G1 归一化动作回报触发。当前适配只支持**盲速度控制**，深度和 LgDog 动作跟踪继续使用 SuperMimic；原版 G1 tracking 使用下一节的原生入口。

上游 README 的 H100 测量不能作为此 PyTorch/LgDog 适配的训练加速结论。

### 检查点、迁移与评估

- `algorithm=xmimic_velocity_bptt`、`training_revision=6`；包含 raw actor、EMA actor、Muon/Adam 状态、EMA 次数、学习率迭代和实际更新计数。
- `policy` 是训练中的 raw actor，`policy_ema` 是平均策略；`deployment_policy` 指定默认加载项。EMA 尚未更新时导出 raw。
- 同一算法可以完整续训，允许调整独立 batch 大小。恢复时 replay/物理现场重建，初始采集会增加物理转移数，不伪装成逐 bit 续跑。
- SuperMimic 与 XMimic-Lg 的完整训练状态不混用。可在相同 XML、history、actor/WM 尺寸下，用 `--init-policy` 跨算法迁移导出策略；WM、critic、优化器和计数重新开始。
- 原版 JAX/G1 检查点不能直接加载到 LgDog。

无需教师即可明确比较 EMA 和 raw：

```bash
python -m supermimic_lg.scripts.evaluate_velocity \
  --checkpoint logs/xmimic_lg_4060/checkpoint.pt \
  --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --policy ema --command 0.4 0 0 --seconds 10 --seeds 0 1 42 \
  --output logs/xmimic_lg_4060/evaluation_ema.json
```

改成 `--policy raw` 测实时策略；默认 `auto` 按检查点选择。报告和观看遥测记录实际算法、实际策略来源，便于确认使用的是哪份权重。

## 原版 XMimic：独立 JAX/G1 环境

未修改的上游运行源码位于 `core/supermimic_lg/vendor/xmimic/`，包含 train.py、xmimic 模块、G1 scene/meta 和 tools。33 个原文件的 SHA-256 位于 `SOURCE_MANIFEST.json`，许可与第三方说明原样保留；图表、实验输出和上游测试未打包。原始来源为用户提供的 `E:/足式机器人/SuperMimic/代码/xmimic-main/xmimic-main` 本地归档，没有已验证的 Git 提交号。

先在现有环境只检查，不导入 JAX、不创建 JAX CUDA 上下文：

```bash
python -m supermimic_lg.scripts.xmimic_native --check
```

原版固定依赖见快照 `requirements.txt`，包括 JAX 0.7.2、MuJoCo/MJX/Warp 3.17.0 等；它与当前 LgDog 环境不同。使用单独环境，并通过原脚本准备 G1 网格与数据：

```bash
conda create -n xmimic-jax python=3.11
conda activate xmimic-jax
cd /home/heartpain/SuperMimic/core/supermimic_lg/vendor/xmimic
python -m pip install -r requirements.txt
bash tools/download_data.sh
python train.py --layout 1GPU --log_dir runs/g1
```

原版动作数据是 **G1 CSV**，不是本工程 LgDog NPZ。网格和动作数据没有随重构包提供。原版默认预分配 90% 显存，遵循其自身配置；源程序使用 `setdefault`，可以在启动前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION` 覆盖它。具体峰值和性能需在实际 GPU 上检查。

也可从 LgDog 环境调用已准备好的独立 Python：

```bash
python -m supermimic_lg.scripts.xmimic_native \
  --python /home/heartpain/miniconda3/envs/xmimic-jax/bin/python \
  --motion-dir /path/to/lafan1_g1 --check

python -m supermimic_lg.scripts.xmimic_native \
  --python /home/heartpain/miniconda3/envs/xmimic-jax/bin/python \
  --motion-dir /path/to/lafan1_g1 \
  -- --layout 1GPU --max_minutes 20 --log_dir runs/g1
```

`--print-command` 只输出 argv；`--check` 检查所选 Python 的依赖版本、源码和数据，不验证 CUDA 运行或训练质量。真正启动前会拒绝未满足这些条件的环境。

## 验证范围

已检查 Muon 分组、Adam 边界参数与 torch Adam 的数值一致性、优化器精确恢复、EMA 的首步复制/后续平均、结构化积分反解、镜像与地形映射、三个种子的 MuJoCo 训练/续训/EMA 加载，以及工作台算法切换和配置保存。

最终完整测试为 **77 passed**；独立安装的 0.3.0 wheel 校验通过全部 33 个上游原文件，包含 LICENSE。Headless Edge 检查了命令预览、算法默认值、配置恢复、4060 起步预设、深度输入限制与 390px 页面宽度。EMA/raw 各完成三个种子、每回合 5 秒的无噪声评估；EMA 检查点完成实际 MuJoCo 回放。详见 [验证记录](validation/2026-11-12_xmimic_validation.json)。

32 步想象训练完成 20 轮真实物理检查：4 环境、16 步初始采集、UTD=0.4，累计 264 条转移，WM/actor/critic 更新 40/20/160，EMA 更新 20。该检查的数据量很小，想象生存率仍低，只用于确认链路执行和数值有限，不代表已学会走路。本机无可用 CUDA，原版 JAX/G1 训练、4060 BF16 性能和收敛未验证。

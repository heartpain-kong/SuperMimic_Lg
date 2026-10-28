# SuperMimic 训练工作台

工作台提供启动训练、停止训练、参数配置、命令预览、实时日志和历史记录。同一工作台同时只启动一个训练任务。它使用启动工作台的 Python 环境，因此应先按 [Ubuntu 22.04 部署说明](UBUNTU_22_04.md) 安装 LgDog/mjlab 和本包。

## Ubuntu 启动

```bash
source "$HOME/work/lgdog-venv/bin/activate"
cd "$HOME/work/SuperMimic_Lg"
git pull
python -m pip install --no-deps -e .
python -m supermimic_lg.workbench --root "$PWD" --port 8089
```

在同一台机器用浏览器打开 `http://127.0.0.1:8089`。远程服务器在你的本机另开终端建立 SSH 隧道，再打开相同地址：

```bash
ssh -N -L 8089:127.0.0.1:8089 用户名@服务器地址
```

SSH 使用自定义端口时加 `-p 端口`。工作台仅监听本机回环地址；隧道两端都用 8089，以匹配 Host 校验。在服务器上用 tmux 运行工作台即可在断开 SSH 后保留训练：

```bash
tmux new -s supermimic
source "$HOME/work/lgdog-venv/bin/activate"
cd "$HOME/work/SuperMimic_Lg"
python -m supermimic_lg.workbench --root "$PWD" --port 8089
```

按 `Ctrl+B` 后按 `D` 离开；`tmux attach -t supermimic` 回到工作台终端。不要同时启动多个工作台来管理同一批训练。

Windows 使用安装了 PyTorch/MuJoCo 的 Python 3.10–3.13 环境，也可运行 `python -m supermimic_lg.workbench --root E:\SuperMimic_Lg --port 8089`。选择 MuJoCo 后端可做真实物理的 CPU 在线 SuperMimic 训练；mjlab CUDA 后端在已部署的 Ubuntu GPU 环境使用。

## 选择入口

| 工作台入口 | 输入 | 使用的训练方式 | 输出 |
|---|---|---|---|
| LgDog · SuperMimic 在线训练 | LgDog 仓库、motion NPZ；无需离线 replay | 仿真采集 → 世界模型 → imagined actor-critic/BPTT | 策略、critic、世界模型和优化器检查点 |
| SuperMimic 世界模型 | replay NPZ：states/actions/contexts/[done] | 拟合可微动力学模型 | 实时损失、world_model.pt |
| LgDog1 动作跟踪 · PPO | LgDog 仓库路径、12 关节/13 body motion NPZ | 现有 mjlab tracking PPO runner | tracking 策略 checkpoint、训练日志 |
| 算法冒烟检查 | LgDog1 或 Go2、CPU/GPU | 合成数据检查 BPTT 梯度 | 维度、损失、动作梯度 |

表单路径是服务器上的路径；可以填绝对路径，也可以填相对于工作目录的路径。tracking 默认 64 个环境、20 次迭代，先确认链路后再扩大规模。GPU 训练前按部署文档检查 PyTorch 和 Warp。进程启动失败时，界面显示“失败”、退出码和完整错误日志，不会将启动成功当成训练成功。

要让 LgDog 真正训练 SuperMimic，请选第一个入口。选择 `mjlab` 使用 GPU 并行物理；选择 `MuJoCo` 使用 CPU 真实物理（界面自动改为 2 个环境和 CPU）。预热轮数内先采集并训练世界模型，之后开始 BPTT 策略更新。`每轮采集步数`、`世界模型更新次数`、`BPTT 更新次数` 控制三部分的更新比例。继续训练时填写同一 motion/XML 和策略结构对应的 `checkpoint.pt`。详细命令见 [在线训练说明](SUPERMIMIC_LGDOG.md)。

参照 HIMLoco 的时间帧堆叠方式，默认使用 **当前 1 帧＋过去 2 帧＝3 帧**。每帧观测独立保留 42 个特征，缓冲区与回放采用 `[环境数,3,42]`，按 `[t-2,t-1,t]` 从旧到新排列，仅在策略网络入口展平为 **126 维**。这不是把 LgDog 动作减少到 1 维，策略仍输出 12 个关节动作。重置时使用首帧填满本环境历史，其他环境不受影响。工作台“观测总帧数（当前＋历史）”默认 3；设为 1 表示仅当前帧。未来目标仍为 10 帧。旧检查点须选择与其一致的总帧数才能继续训练；输入长度不匹配时明确报错。

## 停止和文件位置

点击“停止训练”后，Ubuntu 先向独立训练进程组发 SIGINT，让世界模型保存最近一次进度；超时后升级到 SIGTERM/SIGKILL，涵盖 worker 子进程。Windows 使用 `taskkill /T /F` 结束进程树，因此只能保留最近一次已写入的 checkpoint。PPO 是否在中断时额外保存取决于上游 runner；已有的 checkpoint 会保留。

工作台正常退出（终端 Ctrl+C 或 SIGTERM）时也会停止它启动的训练。浏览器关闭和 SSH 隧道断开不会触发停止。服务器断电、kill -9 等强制退出不保证清理；重启工作台后旧的未完成记录会标为“状态未知”，不能通过旧 PID 自动接管或结束进程。

每次启动生成独立目录：

```text
SuperMimic_Lg/logs/workbench/<训练编号>/
├── job.json          # 参数命令、PID、时间和退出状态
├── train.log         # stdout/stderr，界面显示末尾 64 KiB
├── checkpoint.pt     # 在线 SuperMimic：policy / critic / wm 与优化器状态
├── metrics.jsonl     # 在线 SuperMimic：真实奖励、WM 损失、策略梯度等
└── world_model.pt    # 世界模型任务生成
```

tracking runner 将 checkpoint 和 TensorBoard 文件写入工作目录的 `logs/rsl_rl/`。这两个日志目录不提交到 Git。网页会保留最近 30 条记录，表单参数保存在当前浏览器本地。

## SuperMimic 和 PPO

SuperMimic 的策略优化通过可微世界模型中的短视野 rollout 做 BPTT，梯度从预测回报穿过动力学模型传到策略。PPO 则用采样轨迹估计优势，通过概率比值与裁剪目标更新策略。SuperMimic 使用 actor/critic 或 GAE，不代表它使用 PPO。

本仓库的 LgDog 在线入口已接通采集→世界模型→BPTT 策略更新，直接调用独立 `train_supermimic` runner。PPO 指另一个 LgDog1 tracking 基线入口。在线训练保存的 checkpoint 包含策略；单独拟合世界模型时生成的 `world_model.pt` 仅包含动力学模型。

论文：[SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480)。

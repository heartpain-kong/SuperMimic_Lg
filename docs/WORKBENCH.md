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

Windows 开发环境也可运行 `python -m supermimic_lg.workbench --root E:\SuperMimic_Lg --port 8089`，用于 CPU 冒烟和世界模型拟合；mjlab GPU tracking 仍在已部署的 Ubuntu 上运行。

## 选择入口

| 工作台入口 | 输入 | 使用的训练方式 | 输出 |
|---|---|---|---|
| SuperMimic 世界模型 | replay NPZ：states/actions/contexts/[done] | 拟合可微动力学模型 | 实时损失、world_model.pt |
| LgDog1 动作跟踪 · PPO | LgDog 仓库路径、12 关节/13 body motion NPZ | 现有 mjlab tracking PPO runner | tracking 策略 checkpoint、训练日志 |
| 算法冒烟检查 | LgDog1 或 Go2、CPU/GPU | 合成数据检查 BPTT 梯度 | 维度、损失、动作梯度 |

表单路径是服务器上的路径；可以填绝对路径，也可以填相对于工作目录的路径。tracking 默认 64 个环境、20 次迭代，先确认链路后再扩大规模。GPU 训练前按部署文档检查 PyTorch 和 Warp。进程启动失败时，界面显示“失败”、退出码和完整错误日志，不会将启动成功当成训练成功。

## 停止和文件位置

点击“停止训练”后，Ubuntu 先向独立训练进程组发 SIGINT，让世界模型保存最近一次进度；超时后升级到 SIGTERM/SIGKILL，涵盖 worker 子进程。Windows 使用 `taskkill /T /F` 结束进程树，因此只能保留最近一次已写入的 checkpoint。PPO 是否在中断时额外保存取决于上游 runner；已有的 checkpoint 会保留。

工作台正常退出（终端 Ctrl+C 或 SIGTERM）时也会停止它启动的训练。浏览器关闭和 SSH 隧道断开不会触发停止。服务器断电、kill -9 等强制退出不保证清理；重启工作台后旧的未完成记录会标为“状态未知”，不能通过旧 PID 自动接管或结束进程。

每次启动生成独立目录：

```text
SuperMimic_Lg/logs/workbench/<训练编号>/
├── job.json          # 参数命令、PID、时间和退出状态
├── train.log         # stdout/stderr，界面显示末尾 64 KiB
└── world_model.pt    # 世界模型任务生成
```

tracking runner 将 checkpoint 和 TensorBoard 文件写入工作目录的 `logs/rsl_rl/`。这两个日志目录不提交到 Git。网页会保留最近 30 条记录，表单参数保存在当前浏览器本地。

## SuperMimic 和 PPO

SuperMimic 的策略优化通过可微世界模型中的短视野 rollout 做 BPTT，梯度从预测回报穿过动力学模型传到策略。PPO 则用采样轨迹估计优势，通过概率比值与裁剪目标更新策略。SuperMimic 使用 actor/critic 或 GAE，不代表它使用 PPO。

本仓库已有世界模型和 imagined actor-critic 核心，但完整的 mjlab 采集→世界模型→BPTT 策略更新循环尚未封装成工作台入口。目前工作台上的 PPO 指 LgDog1 tracking 桥接复用的上游 runner。世界模型 checkpoint 不是可直接部署到机器人的策略。

论文：[SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480)。

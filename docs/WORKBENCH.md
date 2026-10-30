# SuperMimic 训练工作台

工作台提供启动训练、停止训练、参数配置、命令预览、实时日志和历史记录。同一工作台同时只启动一个训练任务。它使用启动工作台的 Python 环境，因此应先按 [Ubuntu 22.04 部署说明](UBUNTU_22_04.md) 安装 LgDog/mjlab 和本包。

也可直接在页面的 **策略观看 · SuperMimic** 面板加载检查点并观看真实仿真画面，操作见下文。

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

**速度指令运控**无需参考 NPZ，使用本体观测、速度指令及在线地形世界模型训练；**统一评估**会对速度策略和已有 HIM/PPO 导出使用相同物理场景、指令、种子和评价标准。详见 [速度模式与评估](VELOCITY_MBRL.md)。原有参考动作跟踪仍是独立任务，旧 tracking 检查点不能直接加载成速度策略。速度检查点可直接在策略观看面板播放，无需 motion，并使用实际选定地形。

| 工作台入口 | 输入 | 使用的训练方式 | 输出 |
|---|---|---|---|
| LgDog · SuperMimic 在线训练 | LgDog 仓库、motion NPZ；无需离线 replay | 仿真采集 → 世界模型 → imagined actor-critic/BPTT | 策略、critic、世界模型和优化器检查点 |
| SuperMimic 世界模型 | replay NPZ：states/actions/contexts/[done] | 拟合可微动力学模型 | 实时损失、world_model.pt |
| LgDog1 动作跟踪 · PPO | LgDog 仓库路径、12 关节/13 body motion NPZ | 现有 mjlab tracking PPO runner | tracking 策略 checkpoint、训练日志 |
| 算法冒烟检查 | LgDog1 或 Go2、CPU/GPU | 合成数据检查 BPTT 梯度 | 维度、损失、动作梯度 |

表单路径是服务器上的路径；可以填绝对路径，也可以填相对于工作目录的路径。tracking 默认 64 个环境、20 次迭代，先确认链路后再扩大规模。GPU 训练前按部署文档检查 PyTorch 和 Warp。进程启动失败时，界面显示“失败”、退出码和完整错误日志，不会将启动成功当成训练成功。

要让 LgDog 真正训练 SuperMimic，请选第一个入口。选择 `mjlab` 使用 GPU 并行物理；选择 `MuJoCo` 使用 CPU 真实物理（界面自动改为 2 个环境和 CPU）。**训练地形**可选平地或 `mjlab 粗糙地形`。粗糙地形复用 LgDog 仓库中的 `lgdog1_rough_terrain_cfg()`，包含平地、上/下台阶、上/下坡、随机起伏和波浪地形；初始难度限制在前两级，升级需至少 50 个控制步、参考平面位移达到 2 米、实际沿参考方向前进至少 2 米且达到参考位移的 70%；物理失败或明显停滞则降级，短片段保持等级。粗糙地形目前只支持 mjlab GPU 后端。预热轮数内先采集并训练世界模型，之后开始 BPTT 策略更新。`每轮采集步数`、`世界模型更新次数`、`BPTT 更新次数` 控制三部分的更新比例。继续训练时填写相同地形、motion/XML 和策略结构对应的 `checkpoint.pt`。详细命令见 [在线训练说明](SUPERMIMIC_LGDOG.md)。

地形生成只提供仿真课程，不会把站立参考动作变成步态。使用粗糙地形时，必须提供与 LgDog 关节/body 顺序一致的移动参考动作；当前 `standing.npz` 仍只训练站立。现阶段 SuperMimic actor 使用本体感觉历史和未来动作目标，没有 terrain height scan，因此这是盲地形训练；泛化到未知楼梯/坡度仍需用多地形、多种子评估。现有策略观看器运行在平地 MuJoCo 上，不能忠实预览粗糙地形检查点。

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

## 在工作台观看策略

1. 刷新工作台，在右侧“策略观看 · SuperMimic”选择已保存的 `checkpoint.pt`，或手动填写服务器上的路径。
2. 工作台从检查点同目录的 `config.json` 自动读取 motion 和机器人路径。移动过文件时，展开“参考动作与机器人路径”重新指定；动作和 XML 的哈希必须匹配训练记录。
3. 点击“开始观看”。策略的历史长度、目标帧数和网络大小从检查点恢复，不使用训练表单当前填写的结构。
4. 使用“暂停/继续”“回到起始帧”“结束观看”，并可调整 0.25×/0.5×/1×/2× 速度，以及侧视/正视/俯视。

观看使用独立 CPU MuJoCo 环境和确定性策略动作，不添加训练探索噪声。画面由 MuJoCo 离屏渲染，再经工作台同一端口传给浏览器；本机与 SSH 隧道均可使用，无需额外开放端口。播放到片段结尾或触发高度/倾斜失败后会重置到起始帧。暂停会停止物理推进；结束观看不会停止训练。关闭浏览器不会结束服务器上的观看进程，需点击“结束观看”或正常退出工作台。

目前支持在线 SuperMimic 的完整检查点；PPO 的 `model_*.pt` 和单独动力学模型 `world_model.pt` 不适用于此观看入口。观看加载的是开始时的策略快照，训练继续产生新检查点后，结束并重新开始观看才能载入新权重。日志和最近画面保存在 `logs/viewer/<观看编号>/`，不提交到 Git；不是视频录制。

默认显示 XML 中的 STL 视觉网格，物理惯性、关节和 PD 参数与 CPU 训练后端一致。缺少 STL 时，可勾选“仅显示物理碰撞模型”。站立测试策略会显示站立/调整姿态，不代表已学会行走。

### Ubuntu 离屏渲染

使用部署文档中的 Python/MuJoCo 环境。Linux 默认在观看子进程选择 `MUJOCO_GL=egl`，无需桌面或 X11。通常需要 `libegl1 libgl1 libglib2.0-0` 和可用 EGL 驱动；这些系统库已包含在 LgDog 安装脚本中。

如果服务器只有 CPU 或 EGL 驱动不可用，可安装 `libosmesa6`，然后在启动工作台前选择软件渲染：

```bash
sudo apt-get install -y libosmesa6
MUJOCO_GL=osmesa python -m supermimic_lg.workbench --root "$PWD" --port 8089
```

渲染初始化失败时页面显示“观看失败”和错误日志；没有图像时不会用静态示意图替代。实时网格渲染、播放/暂停/重置和进程隔离已在本机 Windows / MuJoCo 3.5.0 上验证，Ubuntu EGL/OSMesa 仍待对应服务器验证。

训练日志的 `transitions` 记录累计真实采集量，`replay_transitions` 记录当前回放容量。新检查点会保存累计量；旧检查点缺少该计数时，从恢复后的采集重新计数。粗糙地形日志另外报告 `terrain/level_mean`、`terrain/promotions` 和 `terrain/demotions`。静止参考动作会在粗糙地形训练启动前报错，避免将站立训练当作地形行走。

# Ubuntu 22.04 部署、运行与启动

本文按两份并列代码仓库部署：本仓库 `SuperMimic_Lg` 提供四足世界模型与 LgDog1 桥接；`unitree_rl_mjlab_lgdog` 提供 mjlab、MuJoCo-Warp、LgDog1 资产和 tracking runner。

## 1. 系统和 GPU 前提

- Ubuntu 22.04 x86_64；安装脚本支持 Python 3.10–3.13。Ubuntu 22.04 自带的 Python 3.10 可用。
- GPU 训练需要 NVIDIA 驱动在宿主机正常工作。登录后先运行 `nvidia-smi`；如果该命令看不到 GPU，先修复平台驱动，再安装 Python 包。
- 安装脚本会安装 LgDog 工程要求的 PyTorch `>=2.7,<2.8`、mjlab 1.2.0、MuJoCo/MuJoCo-Warp 3.5.0 和 Warp 1.12.0。不要在同一个环境中再随意升级这些底层包。
- 安装会占用数 GB 磁盘空间。租赁平台有持久化数据盘时，将仓库、虚拟环境和 pip 缓存放到数据盘。

## 2. 克隆两个仓库

把下面的 `WORK_ROOT` 改为服务器上有足够空间的持久化目录。GPUFree 示例可用 `/root/gpufree-data`。

```bash
export WORK_ROOT="$HOME/work"
mkdir -p "$WORK_ROOT"
cd "$WORK_ROOT"
git clone https://github.com/heartpain-kong/SuperMimic_Lg.git
git clone https://github.com/heartpain-kong/unitree_rl_mjlab_lgdog.git
```

## 3. 安装 LgDog/mjlab GPU 环境

从包含 `install_ubuntu.sh`、`setup.py`、`lgdog.py` 和 `src/` 的 LgDog 仓库根目录安装：

```bash
cd "$WORK_ROOT/unitree_rl_mjlab_lgdog"
bash install_ubuntu.sh --env-dir "$WORK_ROOT/lgdog-venv"
source "$WORK_ROOT/lgdog-venv/bin/activate"
```

安装脚本会安装系统库、创建隔离环境、安装锁定的 mjlab/MuJoCo/Warp 依赖，并检查任务注册。若镜像已经有脚本所需系统库但当前账号没有 sudo 权限，可加 `--skip-system-deps`。不要在 GPU 训练环境使用 `--cpu`；该选项安装 CPU 版 PyTorch，仅适合导入检查。

安装 SuperMimic 包时使用同一个虚拟环境，并关闭依赖重解析，以保留 mjlab 工程固定的 torch 版本：

```bash
export SM_ROOT="$WORK_ROOT/SuperMimic_Lg"
export MJLAB_ROOT="$WORK_ROOT/unitree_rl_mjlab_lgdog"
source "$WORK_ROOT/lgdog-venv/bin/activate"
python -m pip install --no-deps --editable "$SM_ROOT"
```

## 4. 检查安装和 CUDA

先运行不依赖 mjlab 的算法 CPU 冒烟：

```bash
supermimic-lg-smoke --robot lgdog1 --steps 3 --device cpu
```

再确认 PyTorch 和 Warp 使用 GPU：

```bash
python - <<'PY'
import torch
import warp as wp

print("torch CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch 没有检测到 CUDA，请先检查 NVIDIA 驱动和 PyTorch 安装")
wp.init()
print("GPU:", torch.cuda.get_device_name(0))
print("Warp device:", wp.get_device("cuda:0"))
PY
```

如果 `nvidia-smi` 正常、PyTorch 也能看到 GPU，但 Warp 报 `CUDA driver not available`、`CUDA error 34` 或 `Invalid device identifier`，先检查 `ldconfig -p | grep libcuda.so.1`。部分容器把 CUDA stub 放在库搜索路径前面，需要按平台实际驱动库位置设置 `LD_LIBRARY_PATH`/`LD_PRELOAD`；不要把 `stubs/libcuda.so` 当成宿主机驱动库。设置后重新运行上面的 Warp 检查，再开始训练。

例如 GPUFree 容器确认真实驱动库位于 `/lib/x86_64-linux-gnu/libcuda.so.1` 时，可在启动训练前设置：

```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
```

这两个路径是 GPUFree 的环境示例；其他 Ubuntu 主机先用 `ldconfig -p` 查实际路径，不要照搬不存在的库路径。

## 5. 准备 LgDog1 参考动作

tracking 任务必须传入一个 LgDog1 参考动作 NPZ。动作数据不随本仓库发布，也不能直接使用 G1 的 29 关节 LAFAN1 文件。NPZ 必须包含以下数组：

| 键 | 形状 | 含义 |
|---|---|---|
| `joint_pos` | `[T, 12]` | 关节位置 |
| `joint_vel` | `[T, 12]` | 关节速度 |
| `body_pos_w` | `[T, 13, 3]` | 世界坐标系 body 位置 |
| `body_quat_w` | `[T, 13, 4]` | 世界坐标系 body 四元数 |
| `body_lin_vel_w` | `[T, 13, 3]` | 世界坐标系 body 线速度 |
| `body_ang_vel_w` | `[T, 13, 3]` | 世界坐标系 body 角速度 |

关节顺序为 `[FR, FL, RR, RL] × [hip, thigh, calf]`；body 顺序为 `trunk`，再按相同腿序排列每条腿的 `hip/thigh/calf`。数据还必须与 LgDog1 XML 的单位、身体定义和控制周期一致。已有相同 NPZ schema 的动作可先用转换入口校验并保存：

```bash
python -m supermimic_lg.data.convert /path/to/input_motion.npz "$WORK_ROOT/lgdog_motion.npz" --fps 50
```

这个命令只校验/重排已有 NPZ，不会把 CSV 或人形动作自动转换成 LgDog 动作。源数据关节或身体顺序不同时，必须先按 LgDog 实际对应关系转换，不能盲目套用默认顺序。

## 6. 启动 LgDog1 tracking 任务

先以小环境数、少迭代数检查环境创建和训练链路：

```bash
source "$WORK_ROOT/lgdog-venv/bin/activate"
cd "$SM_ROOT"
python -m supermimic_lg.scripts.train_lgdog \
  --mjlab-root "$MJLAB_ROOT" \
  --motion-file "$WORK_ROOT/lgdog_motion.npz" \
  --env.scene.num-envs=64 \
  --agent.max-iterations=20 \
  --agent.logger=tensorboard \
  --agent.run-name=lgdog_tracking_smoke
```

确认日志显示选中了 CUDA、motion 文件路径正确、环境能完成初始化并产生训练迭代后，再逐步增大 `--env.scene.num-envs` 和 `--agent.max-iterations`。显存不足时降低环境数。日志和 checkpoint 写入 LgDog 工程的 `logs/` 下；确保这个目录位于持久化磁盘。

长时间训练可在 `tmux` 中运行：

```bash
sudo apt-get install -y tmux
tmux new -s lgdog-train
# 在 tmux 会话中执行上面的训练命令
```

按 `Ctrl+B`，再按 `D` 可退出会话而不停止训练；回连使用 `tmux attach -t lgdog-train`。GPU 监控使用 `watch -n 1 nvidia-smi`。

## 7. 启动 SuperMimic 世界模型训练

世界模型入口读取的是仿真交互 replay，而不是上一节的参考动作文件。文件包含实际执行动作对应的状态转移：

- `states`: `[T+1, 37]` 或 `[T+1, N, 37]`；
- `actions`: `[T, 12]` 或 `[T, N, 12]`；
- `contexts`: `[16]` 或 `[N, 16]`；
- 可选 `done`: `[T]` 或 `[T, N]`。

数据中的动作应是控制器实际执行的动作。准备好 `replay.npz` 后运行：

```bash
supermimic-lg-train-wm "$WORK_ROOT/replay.npz" \
  --robot lgdog1 \
  --device cuda:0 \
  --steps 1000 \
  --batch-size 256 \
  --horizon 8
```

世界模型入口会输出训练进度，默认每 100 步及结束时将模型、优化器和步数保存到 `logs/world_model/model.pt`，可用 `--output /path/to/model.pt` 改路径。Linux 收到正常停止信号（SIGINT）时保存最近模型；强制终止时使用最近一次已保存的文件。此文件是动力学模型，还需要策略训练才能得到机器人控制策略。当前尚未提供 mjlab replay 自动采集器，也没有完整 SuperMimic actor-critic 训练 runner。第 6 节的 `train_lgdog` 使用 mjlab PPO 参考动作 tracking runner。

需要通过浏览器启动和结束训练时，按 [工作台说明](WORKBENCH.md) 启动网页控制台。

目前 mjlab 桥接注册的任务是 `LgDog1-SuperMimic-Tracking`。Unitree Go2 配置可用于通用核心和 CPU 冒烟，但本仓库尚未提供 Go2 的 mjlab 任务注册桥。

## 8. 常见问题

- `For tracking tasks, --motion-file must be set`：启动命令缺少 motion 文件参数，或任务配置没有收到该参数。
- `Motion file not found`：检查 NPZ 的绝对路径以及运行账号是否有读权限。
- `expected project source at .../src`：确认 `MJLAB_ROOT` 指向包含 `src/` 的 LgDog 仓库根目录，不要指向其上层父目录。
- `ModuleNotFoundError: mjlab`：使用 LgDog 安装脚本创建的虚拟环境，并在同一个环境内安装本仓库。
- CUDA/Warp 初始化失败：先分别检查 `nvidia-smi`、`torch.cuda.is_available()` 和 `wp.get_device("cuda:0")`，不要在 GPU 尚未通过 Warp 检查时启动长训练。

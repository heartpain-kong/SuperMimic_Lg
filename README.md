# SuperMimic quadruped / LgDog1 复现

这个目录把 `E:\足式机器人\SuperMimic\代码\SuperMimic-Reproduced_CuboiLeo` 的核心思路整理成可复用的四足实现：学习一个带上下文条件的可微动力学模型，用软终止概率保留 BPTT 梯度，再在模型中进行短视野想象更新。状态、动作、目标和跟踪奖励都按机器人维度生成，默认提供 Unitree Go2 与现有 `E:\mjlab\unitree_rl_mjlab_lgdog` 的 LgDog1 映射。

Ubuntu 22.04 的从克隆、GPU/依赖安装到启动步骤见 [`docs/UBUNTU_22_04.md`](docs/UBUNTU_22_04.md)。

安装完成后可用网页启动/停止训练、设置参数、预览命令和查看实时日志：

```bash
python -m supermimic_lg.workbench --root /path/to/SuperMimic_Lg --port 8089
```

本机浏览器打开 `http://127.0.0.1:8089`；Ubuntu 远程服务器使用 SSH 转发后访问。完整步骤见 [工作台说明](docs/WORKBENCH.md)。选择 **LgDog · SuperMimic 在线训练**，可直接进入真实仿真采集、世界模型拟合和 BPTT 策略更新循环；PPO tracking 保留为独立基线。

工作台右侧提供 **策略观看 · SuperMimic**：选择检查点后显示真实 MuJoCo 网格画面，支持暂停/继续、重置、速度与视角切换，观看与训练分别控制。

真正的 LgDog SuperMimic 训练命令、后端和检查点说明见 [LgDog 在线 SuperMimic](docs/SUPERMIMIC_LGDOG.md)。该循环已用实际 LgDog XML 在原生 MuJoCo CPU 上验证；mjlab CUDA 并行后端仍需 Ubuntu GPU 实机验证。

现有 LgDog1 XML 的关节顺序是 `[FR, FL, RR, RL] × [hip, thigh, calf]`，动作缩放为 `0.25`，默认站姿为 `(hip=0, thigh=0, calf=0.2)`。这些值来自 `E:\mjlab\unitree_rl_mjlab_lgdog\unitree_rl_mjlab_lgdog\src\assets\robots\lgdog1\lgdog1_constants.py` 与 `lgdog1.xml`，集中在 [`core/supermimic_lg/config.py`](E:/SuperMimic_Lg/core/supermimic_lg/config.py)。

先安装依赖并跑 CPU 冒烟：

```powershell
cd E:\SuperMimic_Lg
uv venv --python 3.14
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
$env:PYTHONPATH = "E:\SuperMimic_Lg\core"
uv run --python .venv\Scripts\python.exe supermimic-lg-smoke --robot lgdog1 --steps 3
uv run --python .venv\Scripts\python.exe -m pytest -q
```

冒烟程序检查四件事：LgDog1 12 关节维度、世界模型状态积分、终止头损失，以及想象 rollout 的动作梯度能经过动力学回到策略动作。默认上下文维度为 `1 + 12 + 3 = 16`，分别对应摩擦、每个关节的编码器偏置和质心偏移。完整训练需要 NVIDIA GPU 和 mjlab 的 Linux 环境；本 Windows 工作区不宣称完成 GPU 训练。

动作文件采用 mjlab 的 NPZ 键：`joint_pos[T,J]`、`joint_vel[T,J]`、`body_pos_w[T,B,3]`、`body_quat_w[T,B,4]`、`body_lin_vel_w[T,B,3]`、`body_ang_vel_w[T,B,3]`，可选 `fps`。转换和校验：

```powershell
$env:PYTHONPATH = "E:\SuperMimic_Lg\core"
python -m supermimic_lg.data.convert input.npz output_lgdog.npz --fps 50
```

启动桥只在调用时导入 mjlab；它复用现有 `make_tracking_env_cfg`、`MotionCommand`、`MotionTrackingOnPolicyRunner`，替换机器人实体、12 个关节动作缩放、LgDog1 身体列表、足端碰撞匹配和 `trunk` 锚点，并在同一进程注册任务。先用小规模验证：

```powershell
cd E:\SuperMimic_Lg
$env:PYTHONPATH = "E:\SuperMimic_Lg\core"
python -m supermimic_lg.scripts.train_lgdog --mjlab-root E:\mjlab --motion-file E:\path\to\lgdog_motion.npz --env.scene.num-envs=64 --agent.max-iterations=20
```

真实 LgDog 训练前需要准备与 XML 身体顺序一致的参考动作，并确认关节单位、IMU 坐标、动作缩放和控制周期。G1 的 29 DoF LAFAN1 文件不能直接喂给 LgDog1；应先重定向或使用四足 gait/teleoperation 数据，再用 [`core/supermimic_lg/data/motion.py`](E:/SuperMimic_Lg/core/supermimic_lg/data/motion.py) 做维度和 NaN 检查。

代码来源和实现边界记录在 [`docs/REPRODUCTION.md`](E:/SuperMimic_Lg/docs/REPRODUCTION.md)。

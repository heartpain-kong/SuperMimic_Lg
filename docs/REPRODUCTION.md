# 复现与适配记录

实现依据是 `E:\足式机器人\SuperMimic\SuperMimic\README.md`、其 `SuperMimic-Reproduced_CuboiLeo` 源码，以及 `E:\mjlab\unitree_rl_mjlab_lgdog\unitree_rl_mjlab_lgdog` 的 tracking 任务和 LgDog1 资产。

SuperMimic 的可迁移部分是：

- 采集带实际执行动作的状态转移窗口；
- 学习 `s_{t+1}=f_ω(s_t,a_t,ξ)`，其中 `ξ` 保留摩擦、编码器偏置和质心偏移等回合上下文；
- 用解析的跟踪奖励和预测终止概率形成可微想象 rollout；
- 对短视野回报通过 BPTT 更新策略，critic 对想象转移做 GAE 拟合。

人形专用部分已去掉：29 DoF 固定切片、G1 FK、G1 身体名字和 LAFAN1 观测维度。通用状态的布局见 [`state.py`](E:/SuperMimic_Lg/core/supermimic_lg/state.py)，当前 LgDog1 状态为 `3+4+12+3+3+12=37` 维，动作是 12 维。

适配点：

- `config.py` 保存机器人关节/身体顺序、默认姿态、限位、动作缩放和周期；
- `adapters/lgdog1.py` 把 mjlab `Entity.data` 转为标准状态；多环境运行时把 `env.scene.env_origins` 传给 `env_origins`，去掉场景偏移；
- `adapters/mjlab_tracking.py` 在运行时加载已有 mjlab checkout 并注册 `LgDog1-SuperMimic-Tracking`；
- `data/motion.py` 校验 mjlab 参考动作 NPZ；
- `training.py` 提供真实 replay 窗口的 world-model 更新入口，`synthetic.py` 提供无仿真依赖的回归 smoke。
- `actor_critic.py` 提供冻结世界模型上的短视野 actor-critic 更新；actor 保留奖励和软生存图，critic 使用同一批 imagined transitions 的 GAE 目标。

需要在 Linux/NVIDIA 上完成的工作是：从 LgDog1 mjlab 环境记录实际执行动作、拟合完整世界模型、做短视野策略训练、再进行 MuJoCo/真实机器人 sim2real 验证。Windows 侧的 CPU smoke 只验证张量维度、损失、有限值和 BPTT 梯度通路。

安全边界：

- G1/LAFAN1 动作文件不做自动重定向，维度不匹配会立即报错；
- `state_from_mjlab` 假设 `root_ang_vel_b` 已是机身坐标系，和现有 mjlab tracking 代码一致；
- 真实控制器的动作限幅、PD 参数、关节排序和 IMU 外参仍以 LgDog1 部署配置为准，不能从 G1 checkpoint 直接继承。

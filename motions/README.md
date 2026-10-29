# LgDog 平地行走参考

`lgdog_walk.npz` 来自已有 V28 HIM/PPO 行走教师在原生 MuJoCo 中的实际执行轨迹，未手动平移机器人或拼接正弦关节轨迹。教师权重不随数据分发。来源哈希、机器人 XML 哈希及检查结果见 `lgdog_walk.report.json`。

- 50 fps，600 帧，12 秒；12 关节、13 个 link，关节/body 顺序与 SuperMimic LgDog 适配器一致。
- 丢弃启动后的 2 秒，再录制 12 秒。命令速度 0.6 m/s；实际前进约 6.400 米，横向漂移约 -0.448 米，实际平均前向速度约 0.534 m/s。
- 机身触地帧数为 0；四足接触占比约 0.597 / 0.515 / 0.475 / 0.570。
- 所有帧位姿和速度由 MuJoCo link 原点读取，通过 XML FK 位姿/四元数/速度一致性检查。
- 种子 0、1、42 仅改变初始关节的微小扰动；三次均通过录制质量门槛。详见 `validation.json`，这不是未知地形上的鲁棒性评估。

## 在 Ubuntu 工作台使用

更新仓库后，将参考动作填写为：

```text
/home/heartpain/SuperMimic/motions/lgdog_walk.npz
```

先选 **平地**，观测总帧数 3、环境数 64、批大小 128、预测步长 8，确认训练曲线和真实行进表现，再扩大规模。更换参考动作时不要继续加载绑定旧 `standing.npz` 的检查点。

这份文件提供平地行走目标，不能作为上下楼梯已成功的证据。当前 runner 在粗糙地形上仍按 motion 的绝对根/body 轨迹跟踪；爬楼、坡道需要进一步处理目标高度与地形对齐和独立通过评估。12 秒轨迹不是可无缝循环的动作，片段结束按 runner 的正常规则重置。

## 从自己的行走策略重新录制

录制工具支持带完整 `lgdog_him_v1` metadata 的 ONNX 教师，自动读取教师自身的历史帧数（本数据教师是 6 帧，与 SuperMimic 学生输入 3 帧互不影响）。必须由可信的 LgDog 工程导出教师，并使用匹配的 XML、观测、关节顺序和 PD 参数。

```bash
python -m pip install onnxruntime
python -m supermimic_lg.scripts.record_reference \
  --teacher /path/to/policy.onnx \
  --mjlab-root /home/heartpain/mjlab/unitree_rl_mjlab_lgdog \
  --output motions/my_lgdog_walk.npz \
  --speed 0.6 --seconds 12 --warmup 2 --seed 42
```

工具拒绝不匹配的观测/关节/控制契约，摔倒、机身触地、无前进、明显跑偏或足端没有交替离地的轨迹不会发布为 NPZ。它只录制教师真实表现，不能把尚未学会行走的策略自动转换为合格步态。

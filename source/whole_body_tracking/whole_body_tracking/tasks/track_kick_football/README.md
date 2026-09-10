# track_kick_football — G1 视觉踢球任务

在 BeyondMimic 式全身动作跟踪框架上构建的**足球射门任务**：给定一段重定向到 G1 的
右脚推射参考动作（阶段一用 `right_kick.npz`，阶段二用裁减版
`right_kick_trimmed.npz`），策略需要根据**带噪声的
球感知**调整动作，把球踢向每局随机采样的目标方位。

任务注册名：

```text
Tracking-KickFootball-Flat-G1-v0
```

核心机制（详细设计见 [PROJECT.md](PROJECT.md)）：

- **虚拟感知**（借鉴 arXiv:2511.03996）：actor 看到的球观测带距离相关噪声、
  ~25 Hz 刷新、~116 ms 延迟、90% 检测概率（丢检置零 + 可见性标志）；
  critic 保留真值（非对称 actor-critic）
- **靶点锚定球 + 可变方位角**（±15° 课程放开到 ±45°）：固定动作不再最优，
  球通道必须被使用（可辨识性）
- **GOAL 奖励 6 项**：进度（防过冲倒扣）/ 触球（限时限次）/ **方向对齐核**
  （出射方向 vs 靶方向，压 aim 散布）/ **速度区间核**（4–5 m/s，猛抽不加分）/
  **逼近**（robot→球距离势场差分，走过头倒扣，防止"球落到身后"）/ 停滞惩罚
- **裁减动作** `motions/kick_football/right_kick_trimmed.npz`（`scripts/trim_npz.py`
  生成，[170:510) 共 6.8 s）：去掉前后长站立段，episode 6.0 s，
  踢球事件密度约 1.67×
- **奖励分组** MOTION / GOAL / REG：阶段二把跟踪奖励压低甚至归零
- **可选时序策略** `--temporal_actor`：1 s 球感知历史 → MLP encoder → 64 维
  latent 拼接进 actor；decoder 从 latent 重建真值球状态（训练期监督）

## 两阶段训练

### 阶段一：盲踢先验（纯跟踪，无球）

用上游 `tracking` 任务学习踢球动作的跟踪先验：

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick.npz \
  --num_envs 2048 --headless \
  --run_name blind_kick_tracking
```

### 阶段二：视觉踢球（本任务）

从阶段一 checkpoint 暖启动，打开球与目标奖励；**同时压低跟踪奖励**，
让策略从"重放动作"过渡到"依据球况踢球"。动作用裁减版（推荐）：

```bash
KICK_MOTION_WEIGHT=1.0 KICK_GOAL_WEIGHT=1.0 \
python scripts/rsl_rl/train.py \
  --task Tracking-KickFootball-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick_trimmed.npz \
  --num_envs 2048 --headless \
  --init_policy_path logs/rsl_rl/g1_flat/<run>/model_XXXX.pt \
  --run_name kick_vision
```

需要短时记忆抗丢检时加 `--temporal_actor`（encoder-decoder 策略）：

```bash
KICK_MOTION_WEIGHT=1.0 KICK_GOAL_WEIGHT=1.0 \
python scripts/rsl_rl/train.py \
  --task Tracking-KickFootball-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick_trimmed.npz \
  --num_envs 2048 --headless --temporal_actor \
  --init_policy_path logs/rsl_rl/g1_flat/<run>/model_XXXX.pt \
  --run_name kick_vision_temporal
```

奖励组缩放（环境变量，无需改代码）：

| 变量 | 作用 | 阶段一 | 阶段二 |
|---|---|---|---|
| `KICK_MOTION_WEIGHT` | 6 项跟踪奖励全局缩放 | 1.0 | **1.0 起步 → 退火到 0**（满权重保动作质量，GOAL 指标稳了再压） |
| `KICK_GOAL_WEIGHT` | 6 项球/目标奖励全局缩放 | 0 | 1.0 |
| `KICK_REG_WEIGHT` | 正则化奖励全局缩放 | 1.0 | 1.0 |

> 注意：`--init_policy_path` 与 `--resume` 互斥。前者做观测通道对齐的权重迁移
> （普通与 temporal 模式均支持），后者要求维度完全一致。

## 播放 / 可视化

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-KickFootball-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick_trimmed.npz \
  --num_envs 1 \
  --load_run <run目录名> --checkpoint model_XXXX.pt
```

> `--load_run` 传运行目录名、`--checkpoint` 只传文件名（不要传全路径）。

可视化中可以看到：机器人、球（每次 reset 在参考触球点附近重摆）、
目标点 marker（`/Visuals/Command/target`）。

## 目录结构

```text
track_kick_football/
├── kick_env_cfg.py     # 场景（含球）、观测、奖励分组、事件、课程
├── config/g1/          # G1 平台配置与任务注册
└── mdp/
    ├── commands.py     # KickMotionCommand：球 spawn、靶点采样、球状态
    ├── observations.py # ball_state_virtual_b+ball_history_b（actor，虚拟感知）
    │                   # ball_state_b+ball_velocity_b（critic，真值）
    ├── rewards.py      # GOAL 组 6 项：progress/contact/alignment/speed/approach/stagnation
    └── events.py       # 关节/COM 随机化（承自上游）
```

详细设计（每一处修改的依据与实现）见 [PROJECT.md](PROJECT.md)。

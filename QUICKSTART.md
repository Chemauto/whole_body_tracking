# 快速开始

从纯动作跟踪到**视觉踢球**的最简流程。三个任务共用一个框架：

| 任务 | 说明 |
|---|---|
| `Tracking-Flat-G1-v0` | 基线全身动作跟踪（无球） |
| `Tracking-KickFootball-Flat-G1-v0` | 视觉踢球：球 + 随机靶点 + 虚拟感知 + 自动退火 |

> 动作数据：`motions/kick_football/`（右脚推射）；LAFAN1 数据在
> `/data/rl_robot/LAFAN1_Retargeting_Dataset/g1/`。踢球任务的详细设计见
> `source/whole_body_tracking/whole_body_tracking/tasks/track_kick_football/README.md` 与 `PROJECT.md`。

## 0. 安装（仅一次）

在已安装 Isaac Lab 的 Python 环境中：

```bash
cd /data/rl_robot/whole_body_tracking
python -m pip install -e source/whole_body_tracking
```

## 1. 纯跟踪（任意动作）

```bash
# csv → npz（LAFAN1 等新动作）
python scripts/csv_to_npz.py \
  --input_file /data/rl_robot/LAFAN1_Retargeting_Dataset/g1/dance1_subject1.csv \
  --input_fps 30 --output_name dance1_subject1 --output_dir ./motions --headless

# 训练
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --motion_file ./motions/dance1_subject1.npz --headless --run_name dance1

# 回放
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 \
  --motion_file ./motions/dance1_subject1.npz
```

## 2. 踢球两阶段

### 阶段一：盲踢先验（纯跟踪，完整动作）

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick.npz \
  --num_envs 2048 --headless --run_name blind_kick_tracking
```

### 阶段二：视觉踢球（裁减动作 + 自动退火）

```bash
python scripts/rsl_rl/train.py --task=Tracking-KickFootball-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick_trimmed.npz \
  --num_envs 2048 --headless \
  --init_policy_path logs/rsl_rl/g1_flat/<盲踢run目录>/model_XXXX.pt \
  --run_name kick_vision
```

**不需要手动调任何奖励权重**：

- 靶点方位角从 iter 0 起变化（±10° → ±15°@2000 → ±45°@6000，课程自动放宽）；
- 跟踪奖励（MOTION 组）由 `anneal_motion_weight` 课程**按踢球精度指标自动退火**
  1.0 → 0.5 → 0.2 → 0.1 → 0，每次降档在日志打印 `[CURRICULUM] MOTION weight -> X ...`；
- 观测自带虚拟感知（距离噪声 / 25 Hz / 116 ms 延迟 / 10% 丢检 + 可见性标志），
  critic 保留真值；
- 断点续训：加 `--resume True --load_run <run目录> --checkpoint model_XXXX.pt`
  （与 `--init_policy_path` 互斥）。

### 验收 / 可视化

```bash
python scripts/rsl_rl/play.py --task=Tracking-KickFootball-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick_trimmed.npz \
  --num_envs 1 --load_run <run目录> --checkpoint model_XXXX.pt
```

> `--load_run` 传运行目录名、`--checkpoint` 只传文件名（不要全路径）。
> 加 `--video --video_length 300 --headless` 可无头录制 mp4。
> `--temporal_actor` 训练的 checkpoint 暂不支持 play。

## 3. 常用工具

```bash
# 裁减动作（去站立段，提高踢球事件密度）
python scripts/trim_npz.py --input motions/kick_football/right_kick.npz \
  --output motions/kick_football/right_kick_trimmed.npz --start 170 --end 510

# tensorboard
tensorboard --logdir logs/rsl_rl --port 6006
```

关键指标（踢球）：`Metrics/motion/kick_accuracy`（踢球方向核 EMA，越高越准）、
`ball_max_speed`（目标 4–5 m/s 带）、`ball_target_dist`、`Episode_Reward/ball_kick_alignment`。

## 环境要求

Isaac Sim 5.1.0 / Isaac Lab 2.3.x / rsl-rl-lib 3.1.2（`env_isaaclab` conda 环境已配好）。
8 GB 显存建议 `--num_envs 2048` 以内。

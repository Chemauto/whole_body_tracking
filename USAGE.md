# 脚本说明 (USAGE)

本仓库所有可运行脚本的分门别类说明。点击脚本名可跳转到源码。完整端到端流程见
[QUICKSTART.md](QUICKSTART.md)，项目总览见 [README.md](README.md)。

> 所有命令默认在已安装 Isaac Lab 的环境（如 `env_isaaclab`）中、仓库根目录下执行。
> 动作相关脚本中，`{motion_name}` 为 `.csv`/`.npz` 的文件名（不含扩展名）。

---

## 一、数据准备 · 动作处理

把原始 `.csv` 参考动作转换成训练用的 `.npz`，并在仿真/纯 matplotlib 中查看。

### [`scripts/csv_to_npz.py`](scripts/csv_to_npz.py)
**作用**：读取重定向后的 `.csv`（base 位姿 + 关节角），通过正向运动学在 Isaac Sim 里
计算每个刚体的最大坐标信息（位姿/速度/加速度），保存为 `.npz`。这是训练前的必备步骤。
- 关键参数：`--input_file`、`--input_fps`、`--output_name`、`--output_dir`（默认 `./motions`）、
  `--output_fps`、`--frame_range`（截取片段）。
- 示例：
  ```bash
  python scripts/csv_to_npz.py \
    --input_file /data/rl_robot/LAFAN1_Retargeting_Dataset/g1/dance1_subject1.csv \
    --input_fps 30 --output_name dance1_subject1 --output_dir ./motions --headless
  ```

### [`scripts/replay_npz.py`](scripts/replay_npz.py)
**作用**：在 Isaac Sim 中回放已转换的 `.npz` 动作（驱动机器人按参考动作运动），
用于在仿真器里肉眼验证转换结果。需要启动 Isaac Sim。
- 关键参数：`--motion_file`。
- 示例：
  ```bash
  python scripts/replay_npz.py --motion_file ./motions/dance1_subject1.npz
  ```

### [`scripts/visualize_motion.py`](scripts/visualize_motion.py)
**作用**：**纯 matplotlib 3D 火柴人动画**，播放 `.npz` 动作。不启动 Isaac Sim，轻量快速，
适合快速查看动作对不对。相机自动跟随 pelvis，左/右/躯干用不同颜色区分，可鼠标旋转视角。
- 关键参数：`--motion_file`、`--fps`、`--every`（跳帧）、`--start`/`--end`、`--elev`/`--azim`、`--no_floor`。
- 示例：
  ```bash
  python scripts/visualize_motion.py --motion_file ./motions/dance1_subject1.npz
  python scripts/visualize_motion.py --motion_file ./motions/dance1_subject1.npz --fps 25 --every 2
  ```

> `replay_npz.py`（仿真里看，带物理模型渲染）与 `visualize_motion.py`（纯骨架，秒开）二选一即可，
> 后者更适合频繁快速检查。

---

## 二、训练

### [`scripts/rsl_rl/train.py`](scripts/rsl_rl/train.py)
**作用**：用 RSL-RL（PPO）训练动作追踪策略。从本地 `.npz` 加载参考动作，
日志和 checkpoint 默认存到 `logs/rsl_rl/<experiment_name>/<时间戳>_<run_name>/`。
- 关键参数：`--task`（如 `Tracking-Flat-G1-v0`）、`--motion_file`、`--num_envs`、
  `--max_iterations`、`--run_name`、`--headless`、`--logger`（默认 `tensorboard`）、`--seed`。
- 示例：
  ```bash
  python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
    --motion_file ./motions/dance1_subject1.npz --headless --run_name dance1
  ```
- 续训：加 `--resume --load_run <run_dir> --checkpoint model_<iter>.pt`。

---

## 三、评估 · 推理

### [`scripts/rsl_rl/play.py`](scripts/rsl_rl/play.py)
**作用**：加载本地 checkpoint 回放（运行）已训练策略，并把策略导出为 `policy.onnx`
（含部署元数据）。模型从 `logs/rsl_rl/<experiment_name>/` 加载。
- 关键参数：`--task`、`--motion_file`（必填）、`--num_envs`、`--load_run`、`--checkpoint`、`--video`。
- 示例（加载最新 checkpoint）：
  ```bash
  python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 \
    --motion_file ./motions/dance1_subject1.npz
  ```
- 指定某次运行/某 checkpoint：
  ```bash
  python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 \
    --motion_file ./motions/dance1_subject1.npz \
    --load_run 2026-08-06_12-00-00_dance1 --checkpoint model_5000.pt
  ```

---

## 四、辅助 · 公共模块

### [`scripts/rsl_rl/cli_args.py`](scripts/rsl_rl/cli_args.py)
**作用**：`train.py` / `play.py` 共用的命令行参数定义与配置更新逻辑（实验名、run 名、
checkpoint 加载、logger 选择等）。**本身不直接运行**，由训练/评估脚本调用。
- 主要可配置项：`--experiment_name`、`--run_name`、`--resume`、`--load_run`、`--checkpoint`、
  `--logger`（`tensorboard`/`neptune`）、`--log_project_name`。

---

## 速查：典型工作流

```
.csv ──csv_to_npz.py──► .npz ──visualize_motion.py──► (肉眼确认)
                       │
                       ├──train.py──► logs/ (checkpoint + policy.onnx)
                       │                   │
                       └──play.py ◄────────┘ (回放策略 + 导出 onnx)
```

更精简的可复制命令见 [QUICKSTART.md](QUICKSTART.md)。

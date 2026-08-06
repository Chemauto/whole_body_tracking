# 快速开始

以默认任务 `Tracking-Flat-G1-v0` 为例，使用本地 LAFAN1（G1）动作数据，从转换到训练、评估的最简流程。

> 动作数据目录：`/data/rl_robot/LAFAN1_Retargeting_Dataset/g1/`（已下载的 40 个 `.csv`）

## 0. 安装（仅一次）

在已安装 Isaac Lab 的 Python 环境中：

```bash
cd /data/rl_robot/whole_body_tracking
python -m pip install -e source/whole_body_tracking
```

## 1. 转换动作：csv → npz

```bash
python scripts/csv_to_npz.py \
  --input_file /data/rl_robot/LAFAN1_Retargeting_Dataset/g1/dance1_subject1.csv \
  --input_fps 30 --output_name dance1_subject1 --output_dir ./motions --headless
```

生成 `./motions/dance1_subject1.npz`。

## 2. 查看动作（可选）

在 Isaac Sim 中播放转换后的 npz，肉眼确认动作正确：

```bash
python scripts/replay_npz.py --motion_file ./motions/dance1_subject1.npz
```

## 3. 训练

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --motion_file ./motions/dance1_subject1.npz --headless --run_name dance1
```

日志与 checkpoint 保存在 `logs/rsl_rl/g1_flat/<时间戳>_dance1/`。

## 4. 评估（回放已训练策略）

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 \
  --motion_file ./motions/dance1_subject1.npz
```

自动加载 `logs/rsl_rl/g1_flat/` 下最新的 checkpoint。

---

换其它动作只需改 `dance1_subject1` 为 `/data/rl_robot/LAFAN1_Retargeting_Dataset/g1/` 下任意 `.csv` 的名字。

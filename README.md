# BeyondMimic 动作追踪代码

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[官方网站]](https://beyondmimic.github.io/)
[[论文]](https://arxiv.org/abs/2508.08241)
[[视频]](https://youtu.be/RS_MtKVIAzY)

## 概述

BeyondMimic 是一个通用的人形机器人控制框架，能够实现高度动态的动作追踪，在真实世界部署中达到业界领先的动作质量，并通过基于引导扩散（guided diffusion）的控制器实现可控的测试时控制。

本仓库涵盖了 BeyondMimic 中的动作追踪训练部分。**你应当能够在 LAFAN1 数据集中训练任意可 sim-to-real（仿真到现实）的动作，而无需调整任何参数**。

关于仿真到仿真（sim-to-sim）和仿真到现实（sim-to-real）的部署，请参考
[motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller)。

### 其他实现

- 在 [mjlab](https://github.com/mujocolab/mjlab) 中有一个 BeyondMimic 的替代复现版本。mjlab 是一个基于 MuJoCo-Warp、采用 Isaac Lab 风格管理器 API 的全新框架，用于强化学习与机器人研究。具体实现请参见[此处](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py)。

## 安装

- 按照[安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装 Isaac Lab v2.1.0。我们推荐使用 conda 安装方式，因为它能简化从终端调用 Python 脚本的过程。

- 在 Isaac Lab 安装目录之外（即不要放在 `IsaacLab` 目录内）单独克隆本仓库：

```bash
# 方式一：SSH
git clone git@github.com:HybridRobotics/whole_body_tracking.git

# 方式二：HTTPS
git clone https://github.com/HybridRobotics/whole_body_tracking.git
```

- 从 GCS 拉取机器人描述文件

```bash
# 进入仓库目录
cd whole_body_tracking
# 将所有文件/目录中出现的 whole_body_tracking 重命名为你的扩展名称（your_fancy_extension_name）
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- 使用已安装 Isaac Lab 的 Python 解释器，安装本库

```bash
python -m pip install -e source/whole_body_tracking
```

## 动作追踪

### 动作预处理

参考动作以本地文件形式管理：先从 `.csv` 转换成 `.npz`，再在训练/回放时通过路径加载。
注意：参考动作应已完成重定向（retargeted），并且只使用广义坐标（generalized coordinates）。

- 收集参考动作数据集（请遵循相应的原始许可证），我们采用与 Unitree 数据集 .csv 相同的命名约定

    - 经过 Unitree 重定向的 LAFAN1 数据集可在 [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) 上获取
    - Sidekicks（侧踢）动作来自 [KungfuBot](https://kungfu-bot.github.io/)
    - Christiano Ronaldo 庆祝动作来自 [ASAP](https://github.com/LeCAR-Lab/ASAP)。
    - 平衡动作（Balance motions）来自 [HuB](https://hub-robot.github.io/)


- 通过正向运动学（forward kinematics），将重定向后的动作转换为包含最大坐标信息（刚体位姿、刚体速度和刚体加速度）的 `.npz` 文件：

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 \
--output_name {motion_name} --output_dir ./motions --headless
```

这会在 `./motions/` 目录下生成 `{motion_name}.npz`。可用 `--output_dir` 指定其它保存目录，
用 `--frame_range START END` 截取动作的某一段（帧索引从 1 开始，含两端）。

- 通过在 Isaac Sim 中回放动作，确认转换结果正确：

```bash
python scripts/replay_npz.py --motion_file=./motions/{motion_name}.npz
```

### 策略训练

- 使用以下命令训练策略（默认使用 TensorBoard 记录日志）：

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--motion_file ./motions/{motion_name}.npz \
--headless --run_name {run_name}
```

### 策略评估

- 使用以下命令回放（运行）已训练的策略。模型会从本地 `logs/rsl_rl/<experiment_name>/` 目录中加载最新 checkpoint：

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 \
--motion_file ./motions/{motion_name}.npz
```

如需指定某次特定的训练运行或某个 checkpoint，可加上 `--load_run {run_dir}` 和 `--checkpoint {model_xxx.pt}`。
每次保存 checkpoint 时，训练 runner 还会自动在 checkpoint 旁导出用于部署的 `policy.onnx`。

## 代码结构

以下是本仓库代码结构的概述：

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  该目录包含定义 BeyondMimic MDP 的原子函数。以下是各函数的功能说明：

    - **`commands.py`**
      命令库，用于根据参考动作和当前机器人状态计算相关变量，并计算误差。包括位姿与速度误差计算、初始状态随机化以及自适应采样。

    - **`rewards.py`**
      实现 DeepMimic 奖励函数及平滑项。

    - **`events.py`**
      实现域随机化（domain randomization）相关项。

    - **`observations.py`**
      实现用于动作追踪和数据采集的观测项。

    - **`terminations.py`**
      实现提前终止（early termination）和超时（timeout）逻辑。

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  包含追踪任务的环境（MDP）超参数配置。

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  包含追踪任务的 PPO 超参数。

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  包含机器人相关的设置，包括 armature 参数、关节刚度/阻尼计算以及动作缩放（action scale）计算。

- **`scripts`**
  包含用于预处理动作数据、训练策略以及评估已训练策略的工具脚本。

这种结构设计旨在确保模块化，方便开发者在扩展本项目时进行导航与维护。

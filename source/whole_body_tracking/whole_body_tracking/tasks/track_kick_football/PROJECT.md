# PROJECT.md — track_kick_football 设计文档

> 本文档详细记录 `track_kick_football` 任务的完整设计：每一处代码修改、背后的依据
> （来自 BeyondMimic 视觉踢球改造文档的准则）、以及两阶段训练策略。
> 基线是上游 `tasks/tracking`（BeyondMimic 式跟踪），本任务是其"追加式"扩展。

---

## 0. 问题定义

上游 `tracking` 是一个**纯跟踪系统**：策略的最优行为完全由参考动作决定，环境中不存
在需要感知的外部客体。把球加进环境后，如果"该往哪踢"是常数（参考动作固定），策略
完全可以忽略球输入、用固定动作完成任务——这就是**盲跑退化**。

本任务的核心设计目标（继承自参考文档的因果原则）：

> 策略只有在"该输入改变最优动作"时才会真正使用该输入。

因此引入**每局随机的目标方位角**，把靶点锚定在球的实际位置前方，使踢球方向成为
逐局变化的量，"记住一个固定动作"在数学上不再最优。

### 可辨识性约束（任务设计的定量准则）

设方位角半宽为 $a$、踢球方向容差为 $\epsilon$：

$$p_{\mathrm{blind}} = \epsilon / a \quad\text{（盲跑命中率天花板，要低）}$$
$$\epsilon \gtrsim \sigma_{\mathrm{aim}} \quad\text{（容差 ≥ 踢球方向散布，否则谁都踢不中）}$$

当前取值：容差 ±15°（0.2618 rad），方位角起步 ±15°、课程终点 ±45°（0.7854 rad）。
盲踢天花板 = 15/45 ≈ 33%。

---

## 1. 两阶段训练策略（关键：跟踪奖励的退场）

### 阶段一：盲踢先验（上游 tracking 任务，无球）

9 项原版奖励全开，学习"会做右脚推射动作"。产出 checkpoint 供阶段二暖启动。

### 阶段二：视觉踢球（本任务）

**跟踪奖励（MOTION 组）必须压低、甚至归零**。原因：

1. **因果竞争**：MOTION 奖励的最优解是"贴参考动作"，GOAL 奖励的最优解是"把球踢向
   靶点"。两者共存时，若 MOTION 占主导，策略只需贴住参考（参考本身就能踢中摆在
   标准点的球），球通道没有梯度来源；
2. **方位角打开后参考失效**：参考动作只包含"直线踢"一条轨迹。当方位角 ±45° 时，
   贴参考 = 永远直线踢 = 方位角命中率天花板 33%。策略必须**偏离参考**才能拿到
   GOAL 奖励，此时 MOTION 奖励是纯粹的对抗力；
3. 但 MOTION 不能在训练一开始就归零——策略还没有任何运动能力，会退化为乱动。
   因此采用**逐步退火**：`KICK_MOTION_WEIGHT` 从 0.2 起步、随训练压到 0。

实现上不需要改任何代码：`RewardsCfg.__post_init__` 读取环境变量做组缩放
（见 §4），阶段切换只是一行 shell 前缀。

---

## 2. 场景与数据

### 2.1 参考动作分析（离线，一次性）

对 `right_kick.npz`（612 帧 @ 50 fps，G1 29 DoF / 30 links）做轨迹分析得到：

| 事实 | 数值 |
|---|---|
| 动作朝向 | **+Y**（世界系），x 为横向 |
| 站立段 | frame 0–190 |
| 上步 | frame ~230（右脚 y 0→0.45） |
| **摆腿踢球** | frame 244–278，右脚前进速度峰值 **6.6 m/s**，y 从 0.43 冲到 1.63 |
| 参考触球帧 | **frame 265**（右脚 z≈0.11 ≈ 球心高度，y≈0.85） |
| 球标准摆位 | `[0.25, 1.2, 0.12]`（RoboNaldo 实测值，动作世界系） |

`kick_step=265` 写入 G1 config，用于 contact 奖励的时间窗门控。

### 2.2 场景加球（`kick_env_cfg.py` · `MySceneCfg`）

新增 `RigidObjectCfg`（IsaacLab 原生 `SphereCfg`）：

- 半径 0.11 m、质量 0.41 kg（足球 4 号球量级）；
- `max_linear_velocity=1000` 等刚体参数按 RoboNaldo 实测值，防止高速射门被
  PhysX 速度上限截断；
- `collision_group=0`：**球与机器人可碰撞**（默认组内碰撞被隔离会导致穿球）；
- 每个 env 一颗球，prim 路径 `{ENV_REGEX_NS}/Ball`。

### 2.3 起点钳制（`KickMotionCommand._resample_command`）

上游的自适应采样会在整段动作里任选起点，但 episode 长 10 s < 动作长 12.24 s，
起点靠后的 episode 将**永远跳过踢球帧**。因此在采样后把起点钳制在动作前
`start_time_fraction=5%`（frame < 30），保证每个 episode 都经历完整踢球序列
（RoboNaldo 的 `start_time_sampling_fraction` 同款机制）。

---

## 3. 球状态管理（`mdp/commands.py` · `KickMotionCommand`）

`KickMotionCommand(MotionCommand)` **继承不改写**原跟踪逻辑，仅追加：

### 3.1 每局重采样（`_spawn_ball_and_target`）

```
球位置 = env_origin + ball_offset[0.25, 1.2, 0.12] + U(横向±0.1) + U(前后±0.1)
速度   = 0（静态球），姿态 = 单位四元数
```

球的摆位锚定**参考动作的触球点**（动作世界系固定值），噪声范围即课程的对象
（起步 ±0.1 m，可逐步放宽到 ±0.25 m）。

```
方位角 az ~ U(-target_azimuth_range, +target_azimuth_range)
靶点 = 球位置 + 2.5 m × rot_z(az)·[0,1,0]        # 动作朝向 +Y，绕 z 旋转
```

**靶点锚定在球**是可辨识性设计的关键：球到靶点恒为 2.5 m，需要踢球的方向恰好等于
采样的方位角；方向逐局变化，固定动作不再最优。

同时复位每局量：`_min_ball_target_dist`、`_progress`、`_contact_paid`、
`ball_max_speed`。

### 3.2 每步更新（`_update_command` 追加）

```python
ball_target_dist    = ||ball_pos_w - target_pos_w||
_min_ball_target_dist = min(_min_ball_target_dist, ball_target_dist)   # 历史最近
_progress           = clamp(ball_init_target_dist - _min_ball_target_dist, min=0)
_progress_delta     = _progress - 上一步的 _progress
```

`_progress` 是**不可逆进度**：球接近靶点时单调增长，越过靶点后不再变化（min 锁定），
`_progress_delta` 是本步增量——这是 reward 的取款口。

### 3.3 观测属性（anchor 系，与上游 `motion_anchor_*_b` 同一坐标系约定）

| 属性 | 维度 | 说明 |
|---|---|---|
| `ball_pos_b` | 3（取 xy 用） | 球在机器人 anchor（torso）系位置 |
| `ball_to_target_dir_b` | 2 | 球→靶单位向量（yaw-only 旋转，去 z） |
| `ball_vel_b` | 2 | 球速（anchor 系），仅 critic 用 |
| `main_foot_pos_w` | 3 | 踢球脚（右 ankle roll）世界位置 |

### 3.4 度量

`ball_speed` / `ball_max_speed` / `ball_target_dist` 写入 metrics，训练曲线直接
可读（文档 5.2 的教训：必须结合 rollout 指标检查奖励设计）。

---

## 4. 观测与奖励

### 4.1 观测追加（`kick_env_cfg.py` · `ObservationsCfg`）

**硬规则：只追加、不替换、不重排。** `ball_state`（4 维）紧插在 `command` 之后：

```text
actor:  [command(58) | ball_state(4) | motion_anchor_pos_b(3) | ... ]   160 → 164
critic: [command(58) | ball_state(4) | ball_velocity(2) | ... ]        286 → 292
```

- `ball_state = [ball_xy(2), unit(球→靶)(2)]`：不加 z（球始终在地面）；
- `ball_velocity`（2 维）**只给 critic**：球被踢前静止，对 actor 无决策信息量且
  真机上难估计；对 critic，踢出后的球速直接决定后续回报（信息价值不对称）；
- 通道顺序固定是权重迁移（§5）的前提：插入位置变了，旧权重的通道语义就错位，
  loss 照样下降但每个通道的含义已经错了。

### 4.2 GOAL 奖励组（`mdp/rewards.py` 追加 3 项）

| 项 | 形式 | 防 exploit 设计 |
|---|---|---|
| `ball_target_progress` | `_progress_delta`（米/步），权重 4.0 | 对历史最近距离取差 + clamp≥0：**完美射门的过冲不倒扣**（抵达付一次钱，滚过头免费） |
| `foot_ball_contact` | 稀疏 0/1，权重 1.0 | 几何接近判定（右脚-球距 < 0.22 m）；每局最多 `paid_steps=3` 次计费；仅在 `time_steps ≤ kick_step+100` 窗口内——防"贴球蹭分" |
| `ball_speed` | `(v−2.5).clamp(0)`，权重 0.2 | 阈值 2.5 m/s：球必须真的被踢动（阈值过低则轻碰也得分） |

### 4.3 奖励分组与全局缩放（`RewardsCfg.__post_init__`）

```text
MOTION 组（6 项跟踪）   × KICK_MOTION_WEIGHT   阶段二压低 → 0
GOAL   组（3 项球/靶）  × KICK_GOAL_WEIGHT     阶段二打开
REG    组（3 项正则）   × KICK_REG_WEIGHT      保持
```

环境变量驱动、无需改代码，等价于参考文档的 `A3_MOTION_WEIGHT` 等机制。这正是
"阶段二跟踪奖励低甚至为 0"的落点。

### 4.4 球物理域随机化（`EventCfg`）

摩擦 0.3–0.8、弹性 0.3–0.8、质量 0.38–0.48 kg（startup 事件）。真机上不同球差异
显著，固定属性会导致策略过拟合单一接触动力学。

---

## 5. 权重迁移（`scripts/rsl_rl/train.py` · `--init_policy_path`）

阶段一 checkpoint 的 actor 首层是 `Linear(160→512)`，阶段二网络是 `Linear(164→512)`，
直接 `load_state_dict` 会维度不匹配；`--resume` 也要求完全一致。因此新增通道对齐迁移：

```python
prefix = 2 × num_joints          # = 58，command 项的维度（由 DoF 推导，不写魔数）

new[:, :58]        = old[:, :58]      # command 通道原位
new[:, 62:]        = old[:, 58:]      # 其余旧通道整体后移 4，语义不变
new[:, 58:62]                     # ball 4 维保持零初始化
```

- 对所有 2D 权重（actor/critic 首层）按此规则搬移；bias 与深层权重维度不变、直接拷贝；
- **normalizer 不迁移**：rsl_rl 3.1.2 的 `policy.state_dict()` 不含 normalizer（非持久
  模块），新运行的统计量从零重新估计。这实际上**规避了参考文档记录的"冻结
  normalizer / 新通道裸奔"隐患**——无需 fade-in 缓解；
- 零初始化 ball 列的含义：迁移完成的一瞬间，新策略的输出与旧策略**完全相同**
  （ball 通道乘零权重），跟踪能力无损继承，之后由 GOAL 奖励逐步塑造。

与 `--resume` 互斥：resume 恢复迭代计数与优化器状态，init 只搬权重、从头计数
（课程也会从 stage 0 重新开始，符合阶段二预期）。

---

## 6. 方位角课程（`kick_env_cfg.py` · `CurriculumCfg`）

```python
widen_target_azimuth(stages = ((0, 0.0), (2000, 0.2618), (6000, 0.7854)))
```

| iteration | 方位角半宽 | 意图 |
|---|---|---|
| 0 | 0 rad | 靶点永远正前方：先学会"稳定踢中直线" |
| 2000 | 15° | 小幅变化：球通道开始有辨识度 |
| 6000 | 45° | 盲踢天花板 15/45≈33%，视觉收益必须显现 |

课程在运行期按名字寻址改写 `command.cfg.target_azimuth_range`（等价于文档 3.6 的
`modify_env_param` 机制）。**注意**：重命名 command 或奖励项而不同步课程地址，只会在
rollout 时暴露——纯静态检查发现不了。

---

## 7. 与上游 `tracking` 的文件级差异总表

| 文件 | 上游 | 本任务 |
|---|---|---|
| `kick_env_cfg.py` | `tracking_env_cfg.py` | 场景+球；观测插 `ball_state_virtual`+`ball_history`(+critic 球真值)；奖励 9→13 项分 3 组；事件+球随机化；课程非空 |
| `mdp/commands.py` | `MotionCommand` | 追加 `KickMotionCommand` 子类（球 spawn/靶点/进度/虚拟感知/历史环/停滞检测）与 `KickMotionCommandCfg` |
| `mdp/observations.py` | 7 项 | 追加 `ball_state_b` / `ball_velocity_b`（critic 真值）与 `ball_state_virtual_b` / `ball_history_b`（actor 感知版） |
| `mdp/rewards.py` | 6+3 项 | 追加 GOAL 组 4 项（progress / contact / speed / stagnation） |
| `mdp/terminations.py` | 4 项 | **不改**（刻意：避免 yaw/tilt 混算误杀等文档 5.2① 类问题） |
| `mdp/events.py` | 2 项 | 不改（球随机化走 isaaclab 内置函数） |
| `config/g1/` | 3 任务注册 | 1 任务 `Tracking-KickFootball-Flat-G1-v0`；humanoid 配置删除 |
| `utils/temporal_kick.py` | — | ③ 时序 encoder-decoder 策略（可选，`--temporal_actor`） |
| `scripts/rsl_rl/train.py` | — | 新增 `--init_policy_path`（§5）与 `--temporal_actor`（§9.3） |

上游 9 项奖励的项名、参数、权重全部原样保留在 MOTION/REG 组内——阶段一语义完全
兼容，对照实验（有无球 / 有无方位角）随时可做。

---

## 8. 虚拟感知、停滞惩罚与时序策略（借鉴 arXiv:2511.03996）

RoboCup 2025 成人组冠军方案（Booster T1，清华+字节 Seed）证明：把**感知的不完美
直接建模进训练**（而非喂真值）是 sim-to-real 视觉踢球的关键。本任务借鉴其三项机制。

### 8.1 虚拟感知系统（①）

**Actor 永远看不到球的真值**。`KickMotionCommand` 内建一个虚拟感知状态机，
参数取自该论文附录（真机实测）：

| 特性 | 模型 | 参数 |
|---|---|---|
| 检测概率 | 每次感知刷新掷骰 | 90%；丢检时位置置零、flag=0 |
| 位置噪声 | 距离相关高斯 | σ = 0.124·d + 0.149 m |
| 更新频率 | 感知时钟 ~25 Hz（控制 50 Hz） | N(25.36, 1.06²) Hz，刷新间保持上次值 |
| 延迟 | 每次刷新采样 | N(116, 18²) ms，量化为控制步，从历史环取旧值 |

实现：`_update_ball_perception()`（每控制步推进）+ 感知环 `BALL_RING_LENGTH=64`
（历史 50 + 延迟余量）。观测接口：

```text
actor:  ball_state_virtual(5) = [估计位置xy, 球→靶方向xy, 可见flag]
        ball_history(50×5)     = 上述估计的 1 s 时间序列（时间顺序）
critic: ball_state(4) + ball_velocity(2) = 真值（非对称 actor-critic）
```

靶方向用**估计球位**计算（球来自检测，靶来自里程计——与论文的感知抽象一致）。
指标 `Metrics/motion/ball_visible` 可直接监控丢检率。

与论文的差异：不建模 FOV 约束与头控（G1 相机固定）；检测概率不随距离衰减
（我们的球始终 <2 m，在论文 7 m 平坦区内）。

### 8.2 停滞惩罚（②）

论文用其防"站桩"。实现于 `_update_stagnation()`：anchor XY 在过去 50 步
（1 s）内位移 < 0.05 m 且 episode 已过窗口 → `_stagnant` 置位，奖励项
`stagnation`（GOAL 组，权重 −1.0）输出 −1。episode 开始的合法静止
（参考动作前 190 帧是站立）由窗口计数自然豁免。

### 8.3 时序 encoder-decoder 策略（③，可选 `--temporal_actor`）

单帧 MLP 在丢检瞬间失明；论文用 50 帧历史 → 64 维 latent + decoder 重建
真值，使 latent 成为隐式状态估计（其估计误差降 46%，丢检 0.3 s 仍能踢中）。
本任务以最小侵入实现于 `utils/temporal_kick.py`：

```text
actor obs = [command(58) | ball_virtual(5) | history(250) | rest(102)]   (415 维)
                          │                    │
                          │            hist_encoder MLP → latent(64)
                          │                    │
actor 输入 = [current(165) ⊕ latent(64)] = 229 维
decoder(仅训练期): latent(64) → 真值球状态(6)   监督目标取自 critic 观测
```

三个组件，全部继承/复用 rsl_rl 3.1.2 原生结构：

1. **`make_temporal_actor_critic(ActorCritic)`**：只重写 `get_actor_obs`——把
   历史块换成 latent。父类按"缩减后维度"构建 actor/critic/normalizer，所有
   原生路径（act / act_inference / update_normalization / evaluate）自动过
   encoder，零改动；
2. **`PPOWithDecoder(PPO)`**：`update()` 先跑原生 PPO（策略梯度自动流经
   encoder），再跑一轮 decoder 辅助 pass（独立 Adam，lr=1e-4，目标为 storage
   里 critic 观测的真值球通道，MSE）——不复制 PPO 内部实现；
3. **`KickTemporalOnPolicyRunner`**：覆盖 `_construct_algorithm` 完成装配，
   维度全部由环境推导（prefix=2×DoF，hist_len=历史长度×5）；`save()` 跳过
   ONNX 导出（exporter 假设扁平 actor，encoder 版导出是待办）。

**权重迁移适配**：temporal 模式的 actor 首层输入是 229 维（current+latent），
`load_appended_checkpoint` 对 `actor.0.weight` 特判：旧 command 通道 → 前
58 列，旧其余通道 → 跳过 ball(5) 块后对齐，ball 与 latent 列零初始化；
critic 及更深层走通用规则。

**注意**：encoder 同时被 PPO 主优化器和辅助优化器更新（两组 Adam 状态），
辅助 lr 小（1e-4）以 PPO 为主导。这是"两步更新"而非逐 minibatch 联合更新，
对 latent 塑形目的等价。

## 9. 已知局限与后续工作

1. **球状态仍是结构化状态输入**（仿真精确几何量），非 RGB 视觉；真机需补检测、
   深度、坐标变换与时延建模（可先加里程计漂移噪声：yaw σ≈0.6°/s）；
2. **real/blind 置换评估未实现**（文档 5.1）：训练稳定后应在 eval 脚本里做跨环境
   置换 + Wilson 95% CI 判定，避免用绝对命中率自欺；
3. normalizer 从零重估（§5 实测 rsl_rl 3.1.2 的 state_dict 不含 normalizer），
   迁移初期统计量未成熟时若训练不稳，可考虑先冻结 normalizer 若干迭代；
4. `foot_ball_contact` 用几何接近判定而非接触力；若出现"路过计费"异常，改用
   `ball_contact_forces` 传感器（场景里球已开 `activate_contact_sensors`）；
5. 没有加 AMP / 时序观测 / 粗糙地形（文档第一层改造）——本任务聚焦第二层；
   跟踪质量不够时再上。

---

## 9. 复现命令速查

```bash
# 阶段一：盲踢先验（无球）
python scripts/rsl_rl/train.py --task Tracking-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick.npz \
  --num_envs 2048 --headless --run_name blind_kick_tracking

# 阶段二：视觉踢球（MOTION 退火 0.2，GOAL 打开）
KICK_MOTION_WEIGHT=0.2 KICK_GOAL_WEIGHT=1.0 \
python scripts/rsl_rl/train.py --task Tracking-KickFootball-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick.npz \
  --num_envs 2048 --headless \
  --init_policy_path logs/rsl_rl/g1_flat/<run>/model_XXXX.pt \
  --run_name kick_vision

# 后期继续压：resume 时同样带上环境变量
KICK_MOTION_WEIGHT=0.0 KICK_GOAL_WEIGHT=1.0 \
python scripts/rsl_rl/train.py --task Tracking-KickFootball-Flat-G1-v0 ... --resume True ...

# 播放
python scripts/rsl_rl/play.py --task Tracking-KickFootball-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick.npz --num_envs 1
```

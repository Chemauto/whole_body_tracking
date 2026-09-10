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
   因此采用**逐步退火**：`KICK_MOTION_WEIGHT` 从 1.0（满权重）起步——先保住动作质量让
   GOAL 奖励在可用的跟踪基础上长出踢球行为，等 `ball_kick_alignment` 稳定、球速进入
   4–5 m/s 带后再依次压 0.5 → 0.2 → 0（一步压 0.2 会让 entropy 推高动作 std、跟踪
   时序漂移、episode 在触球帧之前被终止，实测验证过这个死循环）。

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
| 参考触球帧 | **frame 265**（右脚 z≈0.11 ≈ 球心高度，y≈1.05） |
| 球标准摆位 | `[0.25, 1.2, 0.12]`（RoboNaldo 实测值，动作世界系） |

`kick_step=265` 写入 G1 config，用于 contact 奖励的时间窗门控。

**裁减版**（阶段二实际使用）：`scripts/trim_npz.py` 切片 `[170:510)` 得到
`right_kick_trimmed.npz`（340 帧 / 6.8 s）——去掉前 3.8 s 与后 2.6 s 的纯站立，
`kick_step=95`（265−170），episode 6.0 s。世界坐标不变（球摆位无需调整）；
踢球事件密度约 1.67×（6.0 s vs 10.0 s episode）。完整版仅用于阶段一先验。

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

**硬规则：只追加、不替换、不重排。** 球观测块紧插在 `command` 之后：

```text
actor:  [command(58) | ball_state_virtual(5) | ball_history(50×5) | 其余(102)]   160 → 415
critic: [command(58) | ball_state真值(4) | ball_velocity(2) | 其余]              286 → 292
```

（虚拟感知的噪声/频率/延迟/丢检建模见 §8.1；时序 encoder-decoder 变体见 §8.3，
其中 actor 首层输入为 current(165)+latent(64)=229。）

- actor 的 `ball_state_virtual = [估计位置xy(2), 球→靶方向xy(2), 可见flag(1)]`：不加 z
  （球始终在地面），值来自 §8.1 的虚拟感知（噪声/25 Hz/延迟/丢检）；
- `ball_history`（50×5）是上述估计的 1 s 时间序列（最旧在前），供 MLP 或 §8.3 的
  时序 encoder 消费；
- `ball_velocity`（2 维）**只给 critic**：球被踢前静止，对 actor 无决策信息量且
  真机上难估计；对 critic，踢出后的球速直接决定后续回报（信息价值不对称）；
- critic 的球通道是**真值**（`ball_state_b` 4 维 + `ball_velocity_b` 2 维），只在
  PrivilegedCfg 注册——非对称 actor-critic；
- 通道顺序固定是权重迁移（§5）的前提：插入位置变了，旧权重的通道语义就错位，
  loss 照样下降但每个通道的含义已经错了。

### 4.2 GOAL 奖励组（`mdp/rewards.py`，6 项）

| 项 | 形式 | 设计意图 / 防 exploit |
|---|---|---|
| `ball_target_progress` | `_progress_delta`（米/步），权重 50（整局满 1.0×2.5m=2.5，对齐论文量级） | 对历史最近距离取差 + clamp≥0：**完美射门的过冲不倒扣**（抵达付一次钱，滚过头免费） |
| `foot_ball_contact` | 稀疏 0/1，权重 10（3 次共 0.6，前期引导信号） | 几何接近判定（右脚-球距 < 0.22 m）；每局最多 `paid_steps=3` 次计费；仅在触球窗口内——防"贴球蹭分" |
| `ball_kick_alignment` | **一次性事件制**：球速上穿 1 m/s 的瞬间付 `exp(−(夹角/15°)²)`，权重 2.5 | **显式方向对齐**：出射方向 vs 球→靶方向，压 σ_aim。必须事件制——逐步计费会让慢滚（更多步超阈值）比标准射门挣得多，与速度带目标相反 |
| `ball_speed` | 区间核：<1 零分，1→4.5 线性爬升，>4.5 高斯衰减(σ=1.5)，权重 0.5 | **速度带控制**：踢了就有分、4–5 m/s 满分、8 m/s 猛抽只值 0.004——参考摆腿物理上能给 6+ m/s，无上界奖励会诱导失控大力抽射 |
| `ball_approach` | 有符号势场差分（robot→球距离逐帧下降），权重 10；**球速>0.5 后冻结** | **保持逼近**：走过头（远离球）倒扣——治理"球落到机器人身后"的失效模式；全局锚点误差在 MOTION 退火后可达 ±0.3 m，没有这项机器人可能游走过球 |
| `stagnation` | 停滞时 −1，权重 −1.0 | anchor 1 s 位移 < 5 cm 即触发，防站桩 |

### 4.3 奖励分组与全局缩放（`RewardsCfg.__post_init__`）

```text
MOTION 组（6 项跟踪）   × KICK_MOTION_WEIGHT   阶段二压低 → 0
GOAL   组（6 项球/靶）  × KICK_GOAL_WEIGHT     阶段二打开
REG    组（3 项正则）   × KICK_REG_WEIGHT      保持
```

环境变量驱动、无需改代码，等价于参考文档的 `A3_MOTION_WEIGHT` 等机制。这正是
"阶段二跟踪奖励低甚至为 0"的落点。

### 4.4 球物理域随机化（`EventCfg`）

摩擦 0.3–0.8、弹性 0.3–0.8、质量 0.38–0.48 kg（startup 事件）。真机上不同球差异
显著，固定属性会导致策略过拟合单一接触动力学。

---

## 5. 权重迁移（`scripts/rsl_rl/train.py` · `--init_policy_path`）

阶段一 checkpoint 的 actor 首层是 `Linear(160→512)`，阶段二是 `Linear(415→512)`
（temporal 变体为 229），直接 `load_state_dict` 会维度不匹配；`--resume` 也要求完全
一致（两者互斥，train.py 显式拒绝同给）。通道对齐迁移规则：

```python
prefix = 2 × num_joints          # = 58，command 项维度（由 DoF 推导，不写魔数）
appended = new_dim − old_dim     # plain: 255 = ball(5)+history(250)

new[:, :58]              = old[:, :58]     # command 通道原位
new[:, 58:58+appended]                     # 球+历史列保持随机初始化
new[:, 58+appended:]     = old[:, 58:]     # 其余旧通道整体后移，语义不变
```

- temporal 特判：actor 首层输入是 `[current(165) | latent(64)]`，旧 102 列映射到
  63..165，ball(5) 与 latent(64) 列保持初始化；critic 与更深层走通用规则；
- **迁移等价性已逐位验证**：把新策略的球+历史通道置零后，actor 输出与旧策略在
  atol 1e-6 内一致（审查代理独立复现）；
- **normalizer**：两个 agents cfg 现已显式 `actor/critic_obs_normalization=True`
  （rsl_rl 3.1.2 的 deprecated `empirical_normalization` 字段因 MISSING→{} 的映射
  缺口是静默 no-op，曾导致全程无归一化）。旧 stage-1 checkpoint 无 normalizer
  统计 → 暖启动后统计从零累积（新通道天然单位初始化，无"裸奔冻结"问题）；
- 新增列保持**随机初始化**（非零）：初始行为 = 旧策略 + 有界扰动，属 net2net
  常规做法。

---

## 6. 方位角课程（`kick_env_cfg.py` · `CurriculumCfg`）

```python
widen_target_azimuth(steps_per_iteration=24, stages = ((0, 0.0), (2000, 0.2618), (6000, 0.7854)))
# iteration = common_step_counter // num_steps_per_env(24) —— 以 PPO 迭代为单位
# （曾以 max_episode_length(300) 为单位，45° 档需 180 万步 > 预算 72 万步，永不开启）
```

| iteration | 方位角半宽 | 意图 |
|---|---|---|
| 0 | 0 rad | 靶点永远正前方：先学会"稳定踢中直线" |
| 2000 | 15° | 小幅变化：球通道开始有辨识度 |
| 6000 | 45° | 盲踢天花板 15/45≈33%，视觉收益必须显现（预算 20% 处开启） |

课程在运行期按名字寻址改写 `command.cfg.target_azimuth_range`（等价于文档 3.6 的
`modify_env_param` 机制）。**注意**：重命名 command 或奖励项而不同步课程地址，只会在
rollout 时暴露——纯静态检查发现不了。

---

## 7. 与上游 `tracking` 的文件级差异总表

| 文件 | 上游 | 本任务 |
|---|---|---|
| `kick_env_cfg.py` | `tracking_env_cfg.py` | 场景+球；观测插 `ball_state_virtual`+`ball_history`(+critic 球真值)；奖励 9→15 项分 3 组（GOAL 6 项）；事件+球随机化；课程非空 |
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
3. temporal 的 aux optimizer（encoder/decoder 的 Adam 动量）不入 checkpoint，
   `--resume` 后从零重启，decoder loss 有一次瞬态抬升；
4. `foot_ball_contact` 用几何接近判定而非接触力；若出现"路过计费"异常，改用
   `ball_contact_forces` 传感器（场景里球已开 `activate_contact_sensors`）；
5. 没有加 AMP / 时序观测 / 粗糙地形（文档第一层改造）——本任务聚焦第二层；
   跟踪质量不够时再上。

---

## 10. 复现命令速查

```bash
# 阶段一：盲踢先验（无球，完整动作）
python scripts/rsl_rl/train.py --task Tracking-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick.npz \
  --num_envs 2048 --headless --run_name blind_kick_tracking

# 阶段二：视觉踢球（裁减动作 + 满权重起步，后续按里程碑退火）
KICK_MOTION_WEIGHT=1.0 KICK_GOAL_WEIGHT=1.0 \
python scripts/rsl_rl/train.py --task Tracking-KickFootball-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick_trimmed.npz \
  --num_envs 2048 --headless \
  --init_policy_path logs/rsl_rl/g1_flat/<run>/model_6000.pt \
  --run_name kick_vision

# 后期继续压：resume 时同样带上环境变量
KICK_MOTION_WEIGHT=0.0 KICK_GOAL_WEIGHT=1.0 \
python scripts/rsl_rl/train.py --task Tracking-KickFootball-Flat-G1-v0 ... --resume True ...

# 播放（--load_run 给目录名，--checkpoint 只给文件名）
python scripts/rsl_rl/play.py --task Tracking-KickFootball-Flat-G1-v0 \
  --motion_file motions/kick_football/right_kick_trimmed.npz --num_envs 1 \
  --load_run <run目录名> --checkpoint model_XXXX.pt
```

from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class KickMotionCommand(MotionCommand):
    """Motion command extended with ball and kick-target state.

    On top of plain motion tracking (inherited unchanged), every episode:
      1. the start time is clamped to the first ``start_time_fraction`` of the
         reference so the kick always happens inside the episode;
      2. the ball is spawned near the reference kick contact point (world
         offset ``ball_offset`` + uniform noise); the reference motion faces +y;
      3. a target is anchored at ``ball + target_distance`` along an azimuth
         sampled from ``+/- target_azimuth_range`` around the motion heading,
         so the required kick direction changes every episode (identifiability);
      4. the running minimum ball-target distance is tracked so goal rewards
         can pay irreversible progress without penalising overshoot.

    The actor never sees the true ball state: a virtual perception system
    (paper arXiv:2511.03996, parameters measured on a real robot) exposes a
    noisy, low-rate, latency-delayed, sometimes-missing ball estimate plus a
    visibility flag and a 1 s history of that estimate. The critic keeps the
    true state (asymmetric actor-critic).
    """

    cfg: KickMotionCommandCfg

    # ring buffer length for perceived ball observations (history + latency headroom)
    BALL_RING_LENGTH = 64

    def __init__(self, cfg: KickMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.ball: RigidObject = env.scene[cfg.ball_name]
        self.main_foot_index = self.robot.body_names.index(cfg.main_foot_name)

        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_azimuth = torch.zeros(self.num_envs, device=self.device)
        self.ball_init_target_dist = torch.full((self.num_envs,), cfg.target_distance, device=self.device)
        self._min_ball_target_dist = torch.full((self.num_envs,), cfg.target_distance, device=self.device)
        self._progress = torch.zeros(self.num_envs, device=self.device)
        self._progress_delta = torch.zeros(self.num_envs, device=self.device)
        self._contact_paid = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # -- virtual perception state --------------------------------------
        # latest perceived ball obs [est_pos_b(2), dir_b(2), visible(1)]
        self._perceived_ball = torch.zeros(self.num_envs, 5, device=self.device)
        self._perception_countdown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._perception_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # ring of perceived obs; index arithmetic in [0, BALL_RING_LENGTH)
        self._ball_ring = torch.zeros(self.num_envs, self.BALL_RING_LENGTH, 5, device=self.device)
        self._ring_step = 0
        self._env_ids = torch.arange(self.num_envs, device=self.device)

        # -- stagnation detection ------------------------------------------
        self._anchor_xy_ring = torch.zeros(self.num_envs, cfg.stagnation_window, 2, device=self.device)
        self._anchor_ring_idx = 0
        self._stagnant = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.metrics["ball_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_max_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_target_dist"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_visible"] = torch.zeros(self.num_envs, device=self.device)

    # -- ball state (world / anchor frame) -----------------------------------

    @property
    def ball_pos_w(self) -> torch.Tensor:
        return self.ball.data.root_pos_w

    @property
    def ball_vel_w(self) -> torch.Tensor:
        return self.ball.data.root_lin_vel_w

    @property
    def ball_pos_b(self) -> torch.Tensor:
        """Ball position in the robot anchor frame (xy is what matters, ball stays on ground)."""
        return quat_apply(quat_inv(self.robot_anchor_quat_w), self.ball_pos_w - self.robot_anchor_pos_w)

    @property
    def ball_to_target_dir_b(self) -> torch.Tensor:
        """Unit xy direction ball -> target in the robot anchor frame."""
        direction = self.target_pos_w - self.ball_pos_w
        direction[..., 2] = 0.0
        direction = torch.nn.functional.normalize(direction, dim=-1, eps=1e-6)
        return quat_apply(quat_inv(yaw_quat(self.robot_anchor_quat_w)), direction)[..., :2]

    @property
    def ball_vel_b(self) -> torch.Tensor:
        return quat_apply(quat_inv(self.robot_anchor_quat_w), self.ball_vel_w)[..., :2]

    @property
    def main_foot_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.main_foot_index]

    # -- per-episode resampling ----------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        # keep the reference kick inside every episode: start near the beginning
        max_start = int(self.cfg.start_time_fraction * (self.motion.time_step_total - 1))
        self.time_steps[env_ids] = torch.clamp(self.time_steps[env_ids], max=max_start)
        self._spawn_ball_and_target(env_ids)

    def _spawn_ball_and_target(self, env_ids: Sequence[int]):
        n = len(env_ids)
        ball_pos = self._env.scene.env_origins[env_ids] + torch.tensor(self.cfg.ball_offset, device=self.device)
        ball_pos[:, 0] += sample_uniform(*self.cfg.ball_lateral_range, (n,), device=self.device)
        ball_pos[:, 1] += sample_uniform(*self.cfg.ball_forward_range, (n,), device=self.device)
        identity_quat = torch.zeros(n, 4, device=self.device)
        identity_quat[:, 3] = 1.0
        zero_vel = torch.zeros(n, 3, device=self.device)
        self.ball.write_root_state_to_sim(
            torch.cat([ball_pos, identity_quat, zero_vel, zero_vel], dim=-1), env_ids=env_ids
        )

        # target anchored at the ball: azimuth sampled around the motion heading (+y)
        azimuth = sample_uniform(-self.cfg.target_azimuth_range, self.cfg.target_azimuth_range, (n,), device=self.device)
        self.target_azimuth[env_ids] = azimuth
        direction = torch.nn.functional.pad(torch.stack([-torch.sin(azimuth), torch.cos(azimuth)], dim=-1), (0, 1))
        self.target_pos_w[env_ids] = ball_pos + self.cfg.target_distance * direction

        self.ball_init_target_dist[env_ids] = self.cfg.target_distance
        self._min_ball_target_dist[env_ids] = self.cfg.target_distance
        self._progress[env_ids] = 0.0
        self._progress_delta[env_ids] = 0.0
        self._contact_paid[env_ids] = 0
        self.metrics["ball_max_speed"][env_ids] = 0.0

        # fresh perception immediately at episode start; seed history with it
        self._perception_countdown[env_ids] = 0
        self._update_ball_perception()
        current = self._perceived_ball
        self._ball_ring[:] = current.unsqueeze(1)
        self._anchor_xy_ring[:] = self.robot_anchor_pos_w[:, None, :2]

    # -- per-step update -------------------------------------------------------

    def _update_command(self):
        super()._update_command()

        # irreversible progress towards the target: pay once on approach, free on overshoot
        ball_target_dist = torch.norm(self.ball_pos_w - self.target_pos_w, dim=-1)
        self._min_ball_target_dist = torch.minimum(self._min_ball_target_dist, ball_target_dist)
        progress = (self.ball_init_target_dist - self._min_ball_target_dist).clamp(min=0.0)
        self._progress_delta = progress - self._progress
        self._progress = progress

        # virtual perception + history ring + stagnation
        self._update_ball_perception()
        self._ball_ring[:, self._ring_step % self.BALL_RING_LENGTH] = self._perceived_ball
        self._ring_step += 1
        self._update_stagnation()

    def _update_ball_perception(self):
        """Advance the virtual perception of the ball (actor-side observation).

        Models four characteristics of onboard vision measured in
        arXiv:2511.03996: detection probability, distance-dependent Gaussian
        noise, reduced update frequency (~25 Hz vs 50 Hz control), and latency.
        Between perception refreshes the last estimate is held; on a miss the
        position is zeroed and the visibility flag drops to 0.
        """
        self._perception_countdown -= 1
        fresh_ids = (self._perception_countdown <= 0).nonzero(as_tuple=False).flatten()
        if fresh_ids.numel() == 0:
            return
        n = fresh_ids.numel()
        cfg = self.cfg

        # next refresh in control steps: N(freq_hz) converted at the control rate
        control_dt = self._env.cfg.decimation * self._env.cfg.sim.dt
        freq_hz = torch.randn(n, device=self.device) * cfg.ball_update_freq_hz[1] + cfg.ball_update_freq_hz[0]
        period = (1.0 / freq_hz / control_dt).round().clamp(min=1)
        # per-env pipeline latency in control steps
        delay_ms = torch.randn(n, device=self.device) * cfg.ball_latency_ms[1] + cfg.ball_latency_ms[0]
        self._perception_delay[fresh_ids] = (delay_ms / 1000.0 / (self._env.cfg.decimation * self._env.cfg.sim.dt)).round().clamp(min=0, max=self.BALL_RING_LENGTH - 2).long()
        self._perception_countdown[fresh_ids] = period.long()

        # detection roll; on miss -> zeros + invisible flag
        detected = torch.rand(n, device=self.device) < cfg.ball_detection_prob

        # distance-dependent Gaussian noise on the perceived position
        distance = torch.norm(self.ball_pos_w[fresh_ids, :2] - self.robot_anchor_pos_w[fresh_ids, :2], dim=-1)
        sigma = cfg.ball_noise_slope * distance + cfg.ball_noise_base
        est_pos_w = self.ball_pos_w[fresh_ids].clone()
        est_pos_w[:, :2] += torch.randn(n, 2, device=self.device) * sigma.unsqueeze(-1)

        # express in the robot anchor frame (position: full rotation, direction: yaw only)
        est_pos_b = quat_apply(quat_inv(self.robot_anchor_quat_w[fresh_ids]), est_pos_w - self.robot_anchor_pos_w[fresh_ids])
        direction = self.target_pos_w[fresh_ids] - est_pos_w
        direction[:, 2] = 0.0
        direction = torch.nn.functional.normalize(direction, dim=-1, eps=1e-6)
        dir_b = quat_apply(quat_inv(yaw_quat(self.robot_anchor_quat_w[fresh_ids])), direction)[:, :2]

        self._perceived_ball[fresh_ids] = 0.0
        visible_rows = detected.nonzero(as_tuple=False).flatten()
        rows = fresh_ids[visible_rows]
        self._perceived_ball[rows, 0:2] = est_pos_b[visible_rows, :2]
        self._perceived_ball[rows, 2:4] = dir_b[visible_rows]
        self._perceived_ball[rows, 4] = 1.0

    def _update_stagnation(self):
        """Flag envs whose anchor barely moved over the trailing window (reward farming guard)."""
        window = self.cfg.stagnation_window
        self._anchor_xy_ring[:, self._anchor_ring_idx] = self.robot_anchor_pos_w[:, :2]
        oldest = self._anchor_xy_ring[:, (self._anchor_ring_idx + 1) % window]
        self._anchor_ring_idx = (self._anchor_ring_idx + 1) % window
        moved = torch.norm(self.robot_anchor_pos_w[:, :2] - oldest, dim=-1)
        self._stagnant = (moved < self.cfg.stagnation_threshold) & (self._env.episode_length_buf > window)

    # -- actor-side (virtual perception) observations --------------------------

    @property
    def ball_obs_delayed(self) -> torch.Tensor:
        """Perceived ball obs [pos(2), dir(2), visible(1)] delayed by the sampled latency."""
        idx = (self._ring_step - 1 - self._perception_delay) % self.BALL_RING_LENGTH
        return self._ball_ring[self._env_ids, idx]

    @property
    def ball_obs_history(self) -> torch.Tensor:
        """Flattened chronological history of perceived ball obs (window x 5)."""
        length = self.cfg.ball_history_length
        idx = (self._ring_step - length + torch.arange(length, device=self.device)) % self.BALL_RING_LENGTH
        return self._ball_ring[:, idx].flatten(1)

    def _update_metrics(self):
        super()._update_metrics()
        speed = torch.norm(self.ball_vel_w, dim=-1)
        self.metrics["ball_speed"] = speed
        self.metrics["ball_max_speed"] = torch.maximum(self.metrics["ball_max_speed"], speed)
        self.metrics["ball_target_dist"] = torch.norm(self.ball_pos_w - self.target_pos_w, dim=-1)
        self.metrics["ball_visible"] = self.ball_obs_delayed[:, 4]

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis and not hasattr(self, "target_visualizer"):
            self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if hasattr(self, "target_visualizer"):
            self.target_visualizer.visualize(self.target_pos_w)


@configclass
class KickMotionCommandCfg(MotionCommandCfg):
    """Configuration for the kick motion command."""

    class_type: type = KickMotionCommand

    ball_name: str = "ball"
    main_foot_name: str = "right_ankle_roll_link"

    # ball spawn: world offset from env origin (reference motion faces +y), plus noise
    ball_offset: tuple[float, float, float] = (0.25, 1.2, 0.12)
    ball_forward_range: tuple[float, float] = (-0.1, 0.1)
    ball_lateral_range: tuple[float, float] = (-0.1, 0.1)

    # target: anchored at the ball, azimuth sampled around the motion heading
    target_distance: float = 2.5
    target_azimuth_range: float = 0.2618  # rad (~15 deg), curriculum widens this

    # episode start time clamp so the reference kick step is always reached
    start_time_fraction: float = 0.05

    # reference time step of ball contact (for time-gating the contact reward)
    kick_step: int = 265

    # -- virtual perception (arXiv:2511.03996 appendix, measured on a real robot) --
    ball_detection_prob: float = 0.9          # P(detect) per perception refresh
    ball_noise_slope: float = 0.124           # sigma = slope * distance + base  [m]
    ball_noise_base: float = 0.149
    ball_update_freq_hz: tuple[float, float] = (25.36, 1.06)   # (mean, std) of perception rate
    ball_latency_ms: tuple[float, float] = (116.0, 18.0)       # (mean, std) of pipeline latency
    ball_history_length: int = 50             # perceived-obs history fed to the actor (1 s @ 50 Hz)

    # -- stagnation penalty --
    stagnation_window: int = 50               # steps (~1 s) over which motion is checked
    stagnation_threshold: float = 0.05        # anchor xy displacement [m] below which env is stagnant

    target_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/target")
    target_visualizer_cfg.markers["frame"].scale = (0.3, 0.3, 0.3)

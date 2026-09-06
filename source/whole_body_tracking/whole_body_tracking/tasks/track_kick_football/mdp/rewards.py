from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.track_kick_football.mdp.commands import KickMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


# ---------------------------------------------------------------------------
# GOAL group: ball / kick-target rewards
# ---------------------------------------------------------------------------


def ball_target_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Irreversible ball progress towards the target, in meters gained this step.

    Progress is measured against the historical minimum ball-target distance and
    clamped at zero, so a hard shot that rolls past the target is never punished
    (the overshoot-payback failure mode): reaching the target pays once, rolling
    beyond is free.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return command._progress_delta.clone()


def foot_ball_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_dist: float = 0.22,
    paid_steps: int = 3,
    window_steps: int = 100,
) -> torch.Tensor:
    """Sparse bonus for the main foot touching the ball near the reference kick moment.

    Geometric proximity test (main foot link to ball center). Credit is capped at
    ``paid_steps`` payments per episode and only granted inside a window around
    the reference kick step, preventing "stand next to the ball" reward farming.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    near = torch.norm(command.main_foot_pos_w - command.ball_pos_w, dim=-1) < contact_dist
    in_window = command.time_steps <= command.cfg.kick_step + window_steps
    fresh = near & in_window & (command._contact_paid < paid_steps)
    command._contact_paid += fresh.long()
    return fresh.float()


def ball_speed(env: ManagerBasedRLEnv, command_name: str, threshold: float = 2.5) -> torch.Tensor:
    """Ball speed beyond ``threshold``: the shot must actually move the ball."""
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return (torch.norm(command.ball_vel_w, dim=-1) - threshold).clamp(min=0.0)


def stagnation_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """-1 while the robot barely moved its anchor over the trailing window.

    Guards against "stand still / stand next to the ball" reward farming
    (arXiv:2511.03996 uses the same guard). Only active after the window has
    fully elapsed within the episode.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return -command._stagnant.float()

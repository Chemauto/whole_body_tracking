from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.tasks.track_kick_football.mdp.commands import KickMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def ball_state_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Ball observation for the actor: [ball_xy(2), unit(ball->target)(2)] in the anchor frame.

    No z channel: the ball always rolls on the ground. Appended (never replacing)
    right after the command term so channel semantics stay aligned for weight transfer.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return torch.cat([command.ball_pos_b[:, :2], command.ball_to_target_dir_b], dim=-1)


def ball_velocity_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Ball xy velocity in the anchor frame, critic-only.

    The actor does not see it: the ball is static until the kick, so it carries no
    decision information; for the critic it directly determines return after the kick.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return command.ball_vel_b


def ball_state_virtual_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Actor-side ball observation through the virtual perception system.

    [est_pos_xy(2), unit(est_ball->target)(2), visible(1)] in the anchor frame.
    The estimate carries distance-dependent noise, ~25 Hz refresh, latency, and
    stochastic misses (zeros + flag 0). The critic keeps the true ball_state_b.
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return command.ball_obs_delayed


def ball_history_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Flattened history of perceived ball obs (window * 5), most recent last.

    Gives the actor short-term memory across perception misses; consumed either
    by the plain MLP or by the temporal encoder (TemporalKickActorCritic).
    """
    command: KickMotionCommand = env.command_manager.get_term(command_name)
    return command.ball_obs_history

"""Visualize a converted motion (.npz) as a 3D stick-figure animation.

Pure numpy + matplotlib — does NOT launch Isaac Sim. It reads the ``body_pos_w``
array recorded by ``csv_to_npz.py`` (world-frame position of every rigid body, per
frame) and animates the G1 humanoid skeleton. Handy for a quick, lightweight check
of whether a retargeted/converted motion looks right, without spinning up the sim.

.. code-block:: bash

    # 基本用法：直接播放
    python scripts/visualize_motion.py --motion_file ./motions/dance1_subject1.npz

    # 放慢一半 + 跳帧加速
    python scripts/visualize_motion.py --motion_file ./motions/walk.npz --fps 25 --every 2

    # 只看某一段（帧区间），并设初始视角
    python scripts/visualize_motion.py --motion_file ./motions/dance.npz --start 0 --end 500 \
        --elev 15 --azim 90

交互：动画循环播放，播放期间可鼠标拖拽旋转视角、滚轮缩放；关闭窗口即退出。
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --------------------------------------------------------------------------- #
# G1 骨架定义
# --------------------------------------------------------------------------- #
# body 顺序必须与 csv_to_npz.py 记录的 body_pos_w 列顺序一致（即 Isaac Sim
# articulation 的 body_names 顺序）。该顺序通过加载 G1 机器人 dump 得到。
G1_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

# child -> parent，从 G1 URDF 的 kinematic tree 确认；pelvis 为根（无父节点）。
G1_PARENT = {
    "left_hip_pitch_link": "pelvis",
    "left_hip_roll_link": "left_hip_pitch_link",
    "left_hip_yaw_link": "left_hip_roll_link",
    "left_knee_link": "left_hip_yaw_link",
    "left_ankle_pitch_link": "left_knee_link",
    "left_ankle_roll_link": "left_ankle_pitch_link",
    "right_hip_pitch_link": "pelvis",
    "right_hip_roll_link": "right_hip_pitch_link",
    "right_hip_yaw_link": "right_hip_roll_link",
    "right_knee_link": "right_hip_yaw_link",
    "right_ankle_pitch_link": "right_knee_link",
    "right_ankle_roll_link": "right_ankle_pitch_link",
    "waist_yaw_link": "pelvis",
    "waist_roll_link": "waist_yaw_link",
    "torso_link": "waist_roll_link",
    "left_shoulder_pitch_link": "torso_link",
    "left_shoulder_roll_link": "left_shoulder_pitch_link",
    "left_shoulder_yaw_link": "left_shoulder_roll_link",
    "left_elbow_link": "left_shoulder_yaw_link",
    "left_wrist_roll_link": "left_elbow_link",
    "left_wrist_pitch_link": "left_wrist_roll_link",
    "left_wrist_yaw_link": "left_wrist_pitch_link",
    "right_shoulder_pitch_link": "torso_link",
    "right_shoulder_roll_link": "right_shoulder_pitch_link",
    "right_shoulder_yaw_link": "right_shoulder_roll_link",
    "right_elbow_link": "right_shoulder_yaw_link",
    "right_wrist_roll_link": "right_elbow_link",
    "right_wrist_pitch_link": "right_wrist_roll_link",
    "right_wrist_yaw_link": "right_wrist_pitch_link",
}


def _bone_color(parent_name: str, child_name: str) -> str:
    """左/右/躯干 用不同颜色，便于区分。"""
    if parent_name.startswith("left") or child_name.startswith("left"):
        return "tab:blue"
    if parent_name.startswith("right") or child_name.startswith("right"):
        return "tab:orange"
    return "black"  # 脊柱：pelvis -> waist -> torso


def build_skeleton(body_names: list[str]) -> tuple[list[tuple[int, int]], list[str]]:
    """由 body_names + G1_PARENT 构建骨头（父索引, 子索引）与每根骨头的颜色。"""
    name_to_idx = {n: i for i, n in enumerate(body_names)}
    edges, colors = [], []
    for child, parent in G1_PARENT.items():
        if child in name_to_idx and parent in name_to_idx:
            edges.append((name_to_idx[parent], name_to_idx[child]))
            colors.append(_bone_color(parent, child))
    if not edges:
        raise RuntimeError("无法构建骨架：npz 的 body_names 与 G1 骨架定义不匹配。")
    return edges, colors


def _floor_grid(center_x: float, center_y: float, half: float, step: float = 0.5):
    """生成 z=0 地面网格的线段（用于每帧跟随 pelvis 重新定位）。"""
    segs = []
    coords = np.arange(center_x - half, center_x + half + step, step)
    for x in coords:
        segs.append([[x, center_y - half, 0.0], [x, center_y + half, 0.0]])
    coords = np.arange(center_y - half, center_y + half + step, step)
    for y in coords:
        segs.append([[center_x - half, y, 0.0], [center_x + half, y, 0.0]])
    return segs


def main():
    parser = argparse.ArgumentParser(description="用 matplotlib 3D 火柴人动画播放 npz 动作。")
    parser.add_argument("--motion_file", type=str, required=True, help="csv_to_npz.py 生成的 npz 路径。")
    parser.add_argument("--fps", type=float, default=None, help="播放帧率（默认用 npz 里的 fps）。")
    parser.add_argument("--every", type=int, default=1, help="帧抽样步长（>1 可加速、降负载）。")
    parser.add_argument("--start", type=int, default=0, help="起始帧（含）。")
    parser.add_argument("--end", type=int, default=-1, help="结束帧（含），-1 表示到末尾。")
    parser.add_argument("--window", type=float, default=1.3, help="xy 视窗半宽（米），相机跟随 pelvis。")
    parser.add_argument("--elev", type=float, default=15.0, help="初始俯仰角。")
    parser.add_argument("--azim", type=float, default=-90.0, help="初始方位角。")
    parser.add_argument("--no_floor", action="store_true", help="不画地面网格。")
    args = parser.parse_args()

    # ---- 加载动作 ----
    data = np.load(args.motion_file, allow_pickle=True)
    if "body_pos_w" not in data.files:
        raise KeyError(f"npz 里没有 body_pos_w 字段：{data.files}")
    body_pos = np.asarray(data["body_pos_w"])  # (T, B, 3)
    body_names = list(data["body_names"]) if "body_names" in data.files else list(G1_BODY_NAMES)

    # 截取帧区间 + 抽样
    end = body_pos.shape[0] if args.end < 0 else min(args.end + 1, body_pos.shape[0])
    start = max(args.start, 0)
    frames = body_pos[start:end:args.every]
    fps = float(args.fps) if args.fps is not None else float(np.asarray(data["fps"]).reshape(-1)[0])
    print(f"[INFO] 动作: {args.motion_file}")
    print(f"[INFO] body 数: {body_pos.shape[1]} | 总帧 {body_pos.shape[0]} | 播放帧 {len(frames)} | fps {fps}")

    edges, edge_colors = build_skeleton(body_names)

    # ---- 画布 ----
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1.1))

    # matplotlib 3.10 的 add_collection3d 对"空"3D 集合会触发 auto_scale_xyz 报错，
    # 因此用首帧数据初始化各集合（update() 每帧会覆盖）。
    _first = frames[0]
    _init_segs = [np.stack([_first[p], _first[c]]) for p, c in edges]
    zmin = min(0.0, float(frames[:, :, 2].min()) - 0.1)
    zmax = float(frames[:, :, 2].max()) + 0.2
    half = args.window

    bones = Line3DCollection(_init_segs, colors=edge_colors, linewidths=2.2)
    ax.add_collection3d(bones)
    (joints,) = ax.plot(_first[:, 0], _first[:, 1], _first[:, 2], "o", color="tab:green", markersize=4)
    if args.no_floor:
        floor = None
    else:
        floor = Line3DCollection(
            _floor_grid(float(_first[0, 0]), float(_first[0, 1]), half), colors="#cccccc", linewidths=0.8
        )
        ax.add_collection3d(floor)

    title = ax.set_title("")

    def init():
        ax.set_zlim(zmin, zmax)
        return bones, joints, title

    def update(i):
        pts = frames[i]  # (B, 3)
        segs = [np.stack([pts[p], pts[c]]) for p, c in edges]
        bones.set_segments(segs)
        joints.set_data(pts[:, 0], pts[:, 1])
        joints.set_3d_properties(pts[:, 2])

        px, py = float(pts[0, 0]), float(pts[0, 1])  # pelvis = body 0
        if floor is not None:
            floor.set_segments(_floor_grid(px, py, half))
        ax.set_xlim(px - half, px + half)
        ax.set_ylim(py - half, py + half)

        t = (start + i * args.every) / fps
        ax.set_title(f"frame {start + i * args.every} / {body_pos.shape[0]}    t = {t:.2f} s")
        return bones, joints, title

    interval = max(1, int(round(1000.0 * args.every / fps)))
    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(frames), interval=interval, blit=False, repeat=True
    )
    ax.view_init(elev=args.elev, azim=args.azim)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

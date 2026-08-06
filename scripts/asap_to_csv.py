"""Convert an ASAP (HumanoidVerse) retargeted motion (.pkl) into the csv format
that whole_body_tracking's csv_to_npz.py expects.

ASAP retargets motions to a 23-DOF "annealed" G1 (it drops the 6 wrist joints).
This script maps those 23 joints back into whole_body_tracking's full 29-DOF G1
joint order (the missing wrists are filled with 0), and writes a 36-column csv:

    [base_x, base_y, base_z, qx, qy, qz, qw, j0, j1, ..., j28]

Both ASAP and the LAFAN csv use (x, y, z, w) quaternions, so root_rot is passed
through unchanged. Verified against:
  humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml

.. code-block:: bash

    # 默认转换 CR7 庆祝动作
    python scripts/asap_to_csv.py \
      --pkl /data/rl_robot/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl \
      --output_name cr7_celebration --output_dir ./motions
"""

import argparse
import os

import numpy as np

# ASAP 23-DOF 顺序（g1_29dof_anneal_23dof.yaml 的 dof_names，已补 _joint）。
# 这正是 pkl 里 dof[:, i] 对应的关节名。
ASAP_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

# whole_body_tracking 的 29-DOF 顺序（与 csv_to_npz.py 记录的 dof 列顺序一致，
# 通过加载 G1 机器人 dump 得到）。ASAP 缺的 6 个手腕关节会被填 0。
WBT_DOF_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# ASAP 缺失、需填默认值的关节。G1 nominal 手腕姿态 ≈ 0。
MISSING_DEFAULT = 0.0


def load_asap_motion(pkl_path: str) -> dict:
    """加载 ASAP pkl，剥掉外层以文件名为 key 的包装。"""
    import joblib

    data = joblib.load(pkl_path)
    if isinstance(data, dict) and "dof" not in data:
        # 形如 { "<filename>": {motion dict} }
        if len(data) == 1:
            data = next(iter(data.values()))
        else:
            # 多段：取第一个含 dof 的
            for v in data.values():
                if isinstance(v, dict) and "dof" in v:
                    data = v
                    break
    if "dof" not in data:
        raise KeyError(f"pkl 里没有 dof 字段，可用键: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    return data


def main():
    parser = argparse.ArgumentParser(description="ASAP pkl -> whole_body_tracking csv。")
    parser.add_argument(
        "--pkl",
        type=str,
        default="/data/rl_robot/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl",
        help="ASAP 重定向后的 motion pkl 路径。",
    )
    parser.add_argument("--output_name", type=str, required=True, help="输出 csv/npz 的文件名（不含扩展名）。")
    parser.add_argument("--output_dir", type=str, default="./motions", help="csv 输出目录。")
    parser.add_argument("--start", type=int, default=0, help="起始帧（含）。")
    parser.add_argument("--end", type=int, default=-1, help="结束帧（含），-1 到末尾。")
    args = parser.parse_args()

    motion = load_asap_motion(args.pkl)
    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)  # (T,3)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)  # (T,4) xyzw
    dof = np.asarray(motion["dof"], dtype=np.float64)  # (T,23)
    fps = int(motion.get("fps", 30)) if not hasattr(motion.get("fps", 30), "shape") else int(np.asarray(motion["fps"]).reshape(-1)[0])
    assert dof.shape[1] == len(ASAP_DOF_NAMES), f"dof 列数 {dof.shape[1]} != {len(ASAP_DOF_NAMES)}"

    # 截取帧区间
    end = root_pos.shape[0] if args.end < 0 else min(args.end + 1, root_pos.shape[0])
    sl = slice(max(args.start, 0), end)
    root_pos, root_rot, dof = root_pos[sl], root_rot[sl], dof[sl]
    T = root_pos.shape[0]

    # 23 dof -> 名字字典
    asap_by_name = {name: dof[:, i] for i, name in enumerate(ASAP_DOF_NAMES)}
    missing = [n for n in WBT_DOF_NAMES if n not in asap_by_name]

    # 映射到 29 dof 顺序（缺失的填默认）
    dof29 = np.stack([asap_by_name.get(n, np.full(T, MISSING_DEFAULT)) for n in WBT_DOF_NAMES], axis=1)

    # 拼成 36 列 csv：base_pos(3) + base_quat xyzw(4) + 29 joints
    csv = np.concatenate([root_pos, root_rot, dof29], axis=1).astype(np.float64)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name + ".csv")
    np.savetxt(out_path, csv, delimiter=",", fmt="%.8f")

    # ---- 诊断信息，方便核对约定是否正确 ----
    print(f"[INFO] 源 pkl: {args.pkl}")
    print(f"[INFO] 帧数: {T} | fps: {fps} | 时长: {T/fps:.2f}s")
    print(f"[INFO] ASAP {len(ASAP_DOF_NAMES)} dof -> 本项目 29 dof；缺失填默认({MISSING_DEFAULT})的关节: {missing}")
    print(f"[INFO] 首帧 base_pos = {root_pos[0]}  (pelvis z 应 ~0.7-0.9m，站立)")
    print(f"[INFO] 首帧 base_quat(xyzw) = {root_rot[0]}  (站立应接近 [0,0,0,1])")
    print(f"[INFO] base z 范围: [{root_pos[:,2].min():.3f}, {root_pos[:,2].max():.3f}] m")
    print(f"[INFO] 已写出 csv: {out_path}  shape={csv.shape}")
    print(f"[INFO] 下一步: python scripts/csv_to_npz.py --input_file {out_path} --input_fps {fps} "
          f"--output_name {args.output_name} --output_dir {args.output_dir} --headless")


if __name__ == "__main__":
    main()

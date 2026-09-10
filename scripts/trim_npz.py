"""Trim a BeyondMimic-format motion NPZ to a frame window.

Slices every per-frame array (joint_pos, joint_vel, body_*_w) to [start:end)
while keeping world coordinates untouched, so ball spawn offsets and anchor
conventions stay valid. Use this to drop long standing segments around a kick
and concentrate episodes on the action-relevant window.

Example:
    python scripts/trim_npz.py \
        --input motions/kick_football/right_kick.npz \
        --output motions/kick_football/right_kick_trimmed.npz \
        --start 170 --end 510
    # reference kick frame 265 -> 265 - 170 = 95 in the trimmed clip
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, help="source motion npz")
parser.add_argument("--output", required=True, help="trimmed motion npz")
parser.add_argument("--start", type=int, required=True, help="first frame kept")
parser.add_argument("--end", type=int, required=True, help="exclusive last frame kept")
args = parser.parse_args()

data = np.load(args.input)
frames = data["joint_pos"].shape[0]
assert 0 <= args.start < args.end <= frames, f"window [{args.start},{args.end}) invalid for {frames} frames"

trimmed = {key: (value[args.start : args.end] if value.shape[0] == frames else value) for key, value in data.items()}
np.savez(args.output, **trimmed)
print(f"[INFO]: trimmed {args.input} [{args.start}:{args.end}) -> {args.output}")
print(f"[INFO]: {frames} -> {trimmed['joint_pos'].shape[0]} frames; subtract {args.start} from any frame index")

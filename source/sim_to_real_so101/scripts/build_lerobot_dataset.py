#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Build a LeRobot dataset from the raw episodes the bridge recorded.

Run with an ISOLATED venv that has lerobot (NOT Isaac's kit-python):
  /workspace/lrenv/bin/python source/sim_to_real_so101/scripts/build_lerobot_dataset.py \
      --raw /workspace/raw_demos \
      --repo_id trenton/bottle_to_basket \
      --root /workspace/datasets/bottle_to_basket \
      --task "put the bottle in the basket"

Mirrors the workshop's LeRobotRecorder exactly (observation.state + action in real arm
units, observation.images.<cam> video), so the result is what GR00T expects.
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINTS = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder the bridge wrote (episode_* + dataset_meta.json)")
    ap.add_argument("--repo_id", required=True, help="e.g. trenton/bottle_to_basket")
    ap.add_argument("--root", required=True, help="output dataset folder")
    ap.add_argument("--task", required=True, help="language task string")
    ap.add_argument("--fps", type=int, default=None, help="override fps (else from dataset_meta.json)")
    ap.add_argument("--push", action="store_true", help="push the finished dataset to the HF Hub (needs huggingface-cli login)")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.raw, "dataset_meta.json")))
    fps = args.fps or int(meta["fps"])
    cams = list(meta["cameras"].keys())
    print(f"[build] fps={fps}  cameras={cams}")

    features = {
        "observation.state": {"dtype": "float32", "fps": fps, "shape": (6,), "names": JOINTS},
        "action": {"dtype": "float32", "fps": fps, "shape": (6,), "names": JOINTS},
    }
    for c in cams:
        h, w = meta["cameras"][c]
        features[f"observation.images.{c}"] = {
            "dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"],
        }

    ds = LeRobotDataset.create(
        args.repo_id, fps=fps, features=features, root=args.root, robot_type="so101_follower",
    )

    def add_frame(frame, task):
        # lerobot 0.4.x: task lives in the frame dict. 0.5.x: task is a kwarg.
        try:
            ds.add_frame({**frame, "task": task})
        except (TypeError, KeyError, ValueError):
            ds.add_frame(frame, task=task)

    ep_dirs = sorted(glob.glob(os.path.join(args.raw, "episode_*")))
    print(f"[build] found {len(ep_dirs)} episode(s)")
    for ep in ep_dirs:
        states = np.load(os.path.join(ep, "observation_state.npy")).astype(np.float32)
        actions = np.load(os.path.join(ep, "action.npy")).astype(np.float32)
        cam_files = {c: sorted(glob.glob(os.path.join(ep, f"images_{c}", "frame_*.png"))) for c in cams}
        T = min(len(states), len(actions), *(len(cam_files[c]) for c in cams)) if cams else len(states)
        for t in range(T):
            frame = {"action": actions[t], "observation.state": states[t]}
            for c in cams:
                frame[f"observation.images.{c}"] = np.asarray(Image.open(cam_files[c][t]).convert("RGB"))
            add_frame(frame, args.task)
        ds.save_episode()
        print(f"[build]   {os.path.basename(ep)}: {T} frames")

    print(f"[build] DONE -> {args.root}")

    if args.push:
        print(f"[build] pushing {args.repo_id} to the HF Hub ...")
        ds.push_to_hub()
        print(f"[build] pushed -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

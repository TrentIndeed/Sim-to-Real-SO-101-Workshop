# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Render the task's cameras to PNG files (no live streaming needed) so you can SEE the
# scene — handy when WebRTC/the viewer won't connect. The desk camera (rgb_external_D455)
# looks at the workspace, so it's exactly the view to check the arm + bottle + basket and
# tune positions. Open the PNGs right in VSCode.
#
#     python source/sim_to_real_so101/scripts/save_scene_views.py \
#         --task Lerobot-So101-Teleop-Bottle-To-Basket
#     # -> /workspace/scene_views/rgb_external_D455.png, rgb_ego.png, ...

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Save the env's camera views to PNG (headless).")
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Bottle-To-Basket")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--frames", type=int, default=8, help="settle/render frames before capture")
parser.add_argument("--out", type=str, default="/workspace/scene_views")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True     # render cameras into the observation
args_cli.headless = True           # no window/stream needed — we save to disk

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import os

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401


def _save(name: str, tensor: torch.Tensor, out_dir: str) -> None:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 4:                       # (num_envs, H, W, C) -> first env
        arr = arr[0]
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.0 + 1e-3:  # floats in [0,1] -> [0,255]
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:    # depth -> grayscale
        arr = arr[..., 0]
    Image.fromarray(arr).save(os.path.join(out_dir, f"{name}.png"))
    print(f"[saved] {name}  {arr.shape}")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    with torch.inference_mode():
        for _ in range(args_cli.frames):    # step a few times so cameras render + settle
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs = env.step(actions)[0]

    os.makedirs(args_cli.out, exist_ok=True)
    visual = obs.get("visual", {}) if isinstance(obs, dict) else {}
    saved = 0
    for name, tensor in visual.items():
        if "rgb" in name or "depth" in name:
            try:
                _save(name, tensor, args_cli.out)
                saved += 1
            except Exception as exc:
                print(f"[skip] {name}: {exc}")
    print(f"\nDone — saved {saved} image(s) to {args_cli.out}. Open them in VSCode "
          f"(rgb_external_D455.png = desk/workspace view).")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

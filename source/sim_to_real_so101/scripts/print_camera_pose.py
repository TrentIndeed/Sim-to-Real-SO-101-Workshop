# SPDX-License-Identifier: Apache-2.0
# Prints the external + ego camera world poses (and robot base) so we can position the
# desk camera precisely instead of guessing. Headless, fast, no streaming.
#   python source/sim_to_real_so101/scripts/print_camera_pose.py
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Bottle-To-Basket")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))

    print("\n================ CAMERA / SCENE POSES (world, meters) ================")
    robot = env.scene["robot"]
    print("robot base pos :", robot.data.root_pos_w[0].cpu().numpy().round(4))
    for name in ["camera_external_D455", "camera_ego"]:
        try:
            cam = env.scene[name]
            pos = cam.data.pos_w[0].cpu().numpy().round(4)
            quat = cam.data.quat_w_world[0].cpu().numpy().round(4) if hasattr(cam.data, "quat_w_world") \
                else cam.data.quat_w_ros[0].cpu().numpy().round(4)
            print(f"{name:22s} pos: {pos}   quat(wxyz): {quat}")
        except Exception as exc:
            print(f"{name}: could not read ({exc})")
    print("======================================================================\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

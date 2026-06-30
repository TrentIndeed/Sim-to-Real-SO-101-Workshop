# SPDX-License-Identifier: Apache-2.0
#
# Teleoperate the Isaac Sim SO-101 with a REAL arm on a *different* machine — your
# follower arm, back-driven by hand as a "leader" (torque off). The arm is on your PC,
# the sim is on the cloud GPU box, so this script receives the arm's joint positions
# over a TCP socket from tools/follower_bridge_sender.py and feeds them into the sim.
#
# IMPORTANT: the cloud box's Isaac kit-python does NOT have `lerobot` installed, so the
# core teleop path here is self-contained (the real->sim joint mapping is a few lines of
# tensor math, copied from LeRobotSO101Interface). `lerobot` is only imported LAZILY,
# and only if you pass --repo_id/--repo_root/--task_name to record a dataset.
#
# Run on the CLOUD box:
#   cd /workspace/isaaclab
#   ./isaaclab.sh -p /workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/scripts/lerobot_agent_bridge.py \
#       --task Lerobot-So101-Teleop-Bottle-To-Basket --livestream 2 \
#       --bind_host 0.0.0.0 --bind_port 5556
# In the viewer: 'R' reset, 'S' start/stop recording, 'C' cancel.
import argparse
import json
import os
import socket
import threading

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab SO-101 teleop over a network bridge (remote real arm).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Bottle-To-Basket")
parser.add_argument("--bind_host", type=str, default=os.getenv("BRIDGE_HOST", "0.0.0.0"))
parser.add_argument("--bind_port", type=int, default=int(os.getenv("BRIDGE_PORT", "5556")))
parser.add_argument("--repo_id", type=str, default=None)
parser.add_argument("--repo_root", type=str, default=None)
parser.add_argument("--task_name", type=str, default=None)
parser.add_argument("--save_mp4", action="store_true", default=False)
parser.add_argument("--depth", action="store_true", default=False)
parser.add_argument("--instance_id_seg", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=101)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.utils.keyboard import KeyboardControl

# ---- self-contained real<->sim joint mapping (no lerobot needed) ----
# Joint order = order in the USD articulation. Ranges (degrees) the USD joints span,
# copied from LeRobotSO101Interface.SO101_USD_MAPPING.
JOINT_ORDER = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]
JOINT_NAMES = [j.split(".")[0] for j in JOINT_ORDER]
SO101_USD_RANGES_DEG = {
    "shoulder_pan": (-110, 110),
    "shoulder_lift": (-100, 100),
    "elbow_flex": (-100, 90),
    "wrist_flex": (-95, 95),
    "wrist_roll": (-160, 160),
    "gripper": (-10, 100),
}


def map_real_to_sim(act_dict, joint_mins, joint_maxs, device):
    """Real arm units -> sim joint targets (radians). Mirrors
    get_raw_actions_tensor + get_mapped_actions_vectorized. Returns (raw, radians)."""
    raw = torch.tensor([act_dict[j] for j in JOINT_ORDER], dtype=torch.float32, device=device)
    normalized = torch.zeros_like(raw)
    normalized[:-1] = (raw[:-1] + 100.0) / 200.0  # first 5 joints: -100..100 -> 0..1
    normalized[-1] = raw[-1] / 100.0              # gripper: 0..100 -> 0..1
    mapped_deg = joint_mins + normalized * (joint_maxs - joint_mins)
    return raw, mapped_deg * torch.pi / 180.0


class ActionServer:
    """Background TCP server. Accepts one sender, reads newline-delimited JSON action
    dicts, keeps only the LATEST. Network rate decoupled from sim rate."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self._latest = None
        self._lock = threading.Lock()
        self.connected = False
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[BRIDGE] Listening for the follower-arm sender on {self.host}:{self.port} ...", flush=True)
        while True:
            conn, addr = srv.accept()
            print(f"[BRIDGE] Sender connected from {addr}. Move the arm by hand to drive the sim.", flush=True)
            self.connected = True
            buf = b""
            try:
                with conn:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    while True:
                        data = conn.recv(4096)
                        if not data:
                            break
                        buf += data
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                with self._lock:
                                    self._latest = json.loads(line.decode("utf-8"))
                            except json.JSONDecodeError as exc:
                                print(f"[BRIDGE] dropped malformed packet: {exc}", flush=True)
            except OSError as exc:
                print(f"[BRIDGE] connection error: {exc}", flush=True)
            self.connected = False
            print("[BRIDGE] Sender disconnected — waiting for reconnect...", flush=True)

    def latest(self):
        with self._lock:
            return self._latest


def main():
    keyboard_control = KeyboardControl()

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO]: Action space: {env.action_space}")
    print(f"[INFO]: 'R' reset | 'S' start/stop recording | 'C' cancel")
    env.reset()
    dev = env.unwrapped.device

    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            ccfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": ccfg.height, "width": ccfg.width}
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")

    joint_mins = torch.tensor([SO101_USD_RANGES_DEG[n][0] for n in JOINT_NAMES], dtype=torch.float32, device=dev)
    joint_maxs = torch.tensor([SO101_USD_RANGES_DEG[n][1] for n in JOINT_NAMES], dtype=torch.float32, device=dev)

    action_server = ActionServer(args_cli.bind_host, args_cli.bind_port)
    actions = torch.zeros(env.action_space.shape, device=dev)

    # --- recording is OPTIONAL and needs lerobot; import it lazily so teleop works without it ---
    recording_mode = all([args_cli.repo_id, args_cli.repo_root, args_cli.task_name])
    iface = recorder = None
    if recording_mode:
        try:
            from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
            from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder
            iface = LeRobotSO101Interface(device=dev, port="", id="bridge", cameras=cameras, fps=30, kind="leader")
            recorder = LeRobotRecorder(
                task_name=args_cli.task_name, repo_id=args_cli.repo_id, dataset_root=args_cli.repo_root,
                fps=30, device=dev, cameras=cameras, save_mp4=args_cli.save_mp4,
                depth=args_cli.depth, instance_id_seg=args_cli.instance_id_seg,
            )
            recorder.init_dataset()
        except ModuleNotFoundError:
            print("[WARNING]: `lerobot` not installed on this box — recording disabled, teleop still works.")
            recording_mode = False
        except ValueError:
            print("[ERROR]: dataset folder already exists — recording disabled.")
            recording_mode = False

    last_dict = None
    warned_wait = False
    step_i = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            act_dict = action_server.latest()
            if isinstance(act_dict, dict) and all(j in act_dict for j in JOINT_ORDER):
                last_dict = act_dict
            elif last_dict is None and not warned_wait:
                print("[BRIDGE] No arm packets yet — sim holding still. Start the local sender.", flush=True)
                warned_wait = True

            raw = None
            if last_dict is not None:
                raw, mapped = map_real_to_sim(last_dict, joint_mins, joint_maxs, dev)
                actions[:] = mapped

            # throttled debug (~ once/sec): proves whether packets arrive AND values move
            step_i += 1
            if step_i % 60 == 0:
                if last_dict is None:
                    print(f"[BRIDGE] rx: NONE (no packets yet)  connected={action_server.connected}", flush=True)
                else:
                    print(f"[BRIDGE] rx pan={last_dict['shoulder_pan.pos']:.1f} "
                          f"grip={last_dict['gripper.pos']:.1f}  connected={action_server.connected}", flush=True)

            obs, _, _, _, _ = env.step(actions)

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                env.reset()
                continue

            if recording_mode and keyboard_control.recording and raw is not None:
                visual_obs = obs.get("visual", None)
                if visual_obs is None:
                    print("[WARNING]: No 'visual' obs group - recording needs cameras")
                    keyboard_control.recording = False
                    continue
                joint_pos_obs = obs["policy"]["joint_pos_obs"][0]
                real_obs, vbuf, dbuf, sbuf = iface.sim_to_real_dataset_processor(joint_pos_obs, obs["visual"])
                recorder.push_frame_to_buffer(raw, real_obs, vbuf, dbuf, sbuf)

    env.close()


if __name__ == "__main__":
    main()
    while True:
        simulation_app.update()

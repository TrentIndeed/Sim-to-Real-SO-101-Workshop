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
import queue
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
parser.add_argument("--raw_out", type=str, default="/workspace/raw_demos",
                    help="Folder to write raw recorded episodes to (converted to a LeRobot "
                         "dataset later by scripts/build_lerobot_dataset.py).")
parser.add_argument("--view_camera", type=str, default="external_cam_D455",
                    help="Camera the MAIN viewport looks through at launch (no manual "
                         "re-selecting each time). Use 'perspective' to keep the free camera.")
parser.add_argument("--view_camera2", type=str, default="Robot/gripper/gripper_cam",
                    help="Camera the SECOND viewport looks through (the gripper/ego cam). "
                         "Empty string to skip the second viewport.")
parser.add_argument("--view_res", type=str, default="1280x960",
                    help="Viewport render resolution WxH (4:3 matches the cameras, no squish).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

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


def sim_rad_to_real(state_rad, joint_mins, joint_maxs):
    """Inverse of map_real_to_sim's value mapping: sim joint radians -> real arm units
    (what a real follower would report). Used for the recorded observation.state."""
    mapped_deg = state_rad * 180.0 / torch.pi
    normalized = (mapped_deg - joint_mins) / (joint_maxs - joint_mins)
    raw = torch.zeros_like(normalized)
    raw[:-1] = normalized[:-1] * 200.0 - 100.0
    raw[-1] = normalized[-1] * 100.0
    return raw


def _rgb_to_uint8(t):
    arr = t.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.0 + 1e-3:   # floats in [0,1] -> [0,255]
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
    return arr


class RawRecorder:
    """Records demos to disk WITHOUT lerobot (it can't go in kit-python without breaking
    Isaac). Each episode -> a folder: observation_state.npy, action.npy, timestamps.npy,
    images_<cam>/frame_%05d.png. Build the LeRobot dataset later with
    scripts/build_lerobot_dataset.py in a separate venv."""

    def __init__(self, out_root, cameras, fps):
        self.out_root = out_root
        self.cam_names = list(cameras.keys())
        self.fps = fps
        os.makedirs(out_root, exist_ok=True)
        self.ep = self._next_index()
        self._reset()
        meta = {
            "fps": fps,
            "cameras": {c: [cameras[c]["height"], cameras[c]["width"]] for c in self.cam_names},
            "state_names": JOINT_ORDER,
            "action_names": JOINT_ORDER,
        }
        with open(os.path.join(out_root, "dataset_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        # write episodes to disk on a background thread so the sim/teleop never freezes
        self._save_queue = queue.Queue()
        threading.Thread(target=self._save_worker, daemon=True).start()

    def _next_index(self):
        i = 0
        while os.path.exists(os.path.join(self.out_root, f"episode_{i:04d}")):
            i += 1
        return i

    def _reset(self):
        self.states, self.actions = [], []
        self.frames = {c: [] for c in self.cam_names}

    @property
    def n(self):
        return len(self.states)

    def add(self, state6, action6, rgb_by_cam):
        self.states.append(state6)
        self.actions.append(action6)
        for c in self.cam_names:
            self.frames[c].append(rgb_by_cam[c])

    def save(self):
        if not self.states:
            print("[RAW] nothing to save (0 frames) — record something first.", flush=True)
            return
        # hand the buffers to the background writer and reset immediately (no freeze).
        # The actual episode number is chosen at WRITE time so we never overwrite.
        self._save_queue.put((self.states, self.actions, self.frames))
        print(f"[RAW] episode ({len(self.states)} frames) queued — writing in background "
              f"(wait for the 'saved' line before shutting down).", flush=True)
        self._reset()

    def _save_worker(self):
        while True:
            states, actions, frames = self._save_queue.get()
            try:
                # re-scan disk and take the first FREE slot -> can never clobber an existing
                # episode (covers async kills, restarts, and reused numbers across instances)
                ep = self._next_index()
                d = os.path.join(self.out_root, f"episode_{ep:04d}")
                os.makedirs(d, exist_ok=True)
                T = len(states)
                np.save(os.path.join(d, "observation_state.npy"), np.asarray(states, dtype=np.float32))
                np.save(os.path.join(d, "action.npy"), np.asarray(actions, dtype=np.float32))
                np.save(os.path.join(d, "timestamps.npy"), (np.arange(T, dtype=np.float32) / float(self.fps)))
                for c in self.cam_names:
                    cdir = os.path.join(d, f"images_{c}")
                    os.makedirs(cdir, exist_ok=True)
                    for i, img in enumerate(frames[c]):
                        Image.fromarray(img).save(os.path.join(cdir, f"frame_{i:05d}.png"), compress_level=1)
                print(f"[RAW] saved episode {ep:04d}: {T} frames, cams={self.cam_names} -> {d}", flush=True)
            except Exception as exc:
                print(f"[RAW] ERROR saving episode {ep:04d}: {exc!r}", flush=True)
            finally:
                self._save_queue.task_done()


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


def _setup_viewports(cam1, cam2, res):
    """Point the main viewport at cam1 (desk) and a second viewport at cam2 (gripper),
    both at the given WxH. Creates the second viewport if it isn't open. Camera names are
    relative to /World/envs/env_0/. Best-effort: never fatal."""
    try:
        vw, vh = (int(x) for x in res.lower().split("x"))
    except Exception:
        vw, vh = 1280, 960

    def _apply(api, cam):
        api.camera_path = f"/World/envs/env_0/{cam}"
        try:
            api.resolution = (vw, vh)
        except Exception:
            pass

    # main viewport -> desk camera
    if cam1 and cam1.lower() != "perspective":
        try:
            from omni.kit.viewport.utility import get_active_viewport
            vp = get_active_viewport()
            if vp is not None:
                _apply(vp, cam1)
                print(f"[VIEW] main viewport -> {cam1} @ {vw}x{vh}", flush=True)
        except Exception as exc:
            print(f"[VIEW] main viewport not set ({exc!r})", flush=True)

    # second viewport -> gripper camera (reuse an existing 2nd one, else create it)
    if cam2:
        try:
            api2 = None
            try:
                from omni.kit.viewport.window import get_viewport_window_instances
                wins = list(get_viewport_window_instances())
                if len(wins) >= 2:
                    api2 = wins[1].viewport_api
            except Exception:
                pass
            if api2 is None:
                from omni.kit.viewport.utility import create_viewport_window
                api2 = create_viewport_window(name="Gripper Cam").viewport_api
            _apply(api2, cam2)
            print(f"[VIEW] second viewport -> {cam2} @ {vw}x{vh}", flush=True)
        except Exception as exc:
            print(f"[VIEW] second viewport not set ({exc!r})", flush=True)


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

    # Auto-configure the livestream viewports (cameras + 4:3 aspect) so they're framed right
    # on every launch — no re-selecting cameras / fixing aspect ratios by hand.
    _setup_viewports(args_cli.view_camera, args_cli.view_camera2, args_cli.view_res)

    action_server = ActionServer(args_cli.bind_host, args_cli.bind_port)
    actions = torch.zeros(env.action_space.shape, device=dev)

    # Raw recorder — no lerobot needed (lerobot can't go in kit-python without breaking
    # Isaac). 'S' start/stop (stop saves the episode), 'R' reset (also saves if recording).
    raw_recorder = RawRecorder(args_cli.raw_out, cameras, fps=30)
    print(f"[RAW] recording episodes to {args_cli.raw_out}. So far {raw_recorder.ep} exist.", flush=True)
    prev_recording = False

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

            # ---- raw recording: buffer frames while recording; save when it stops ----
            recording = keyboard_control.recording
            if recording and raw is not None:
                visual_obs = obs.get("visual", None)
                if visual_obs is not None:
                    state_real = sim_rad_to_real(obs["policy"]["joint_pos_obs"][0], joint_mins, joint_maxs)
                    rgb = {c: _rgb_to_uint8(visual_obs[f"rgb_{c}"][0]) for c in cameras}
                    raw_recorder.add(state_real.cpu().numpy(), raw.cpu().numpy(), rgb)
            if prev_recording and not recording:   # 'S' (or 'R') stopped recording -> save
                raw_recorder.save()
            prev_recording = recording

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                env.reset()
                continue

    env.close()


if __name__ == "__main__":
    main()
    while True:
        simulation_app.update()

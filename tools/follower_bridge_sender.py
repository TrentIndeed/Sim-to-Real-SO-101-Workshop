"""Stream your SO-101 follower arm's joint positions to the cloud Isaac Sim teleop.

Run this on the machine the ARM is plugged into (your Windows PC, COM3) with your
lerobot venv. It opens the arm as a "leader" (torque OFF, so you can back-drive it by
hand) and streams its joint positions over TCP to lerobot_agent_bridge.py running on
the cloud GPU box. Move the arm by hand -> the sim arm mirrors it.

    python tools/follower_bridge_sender.py --host <cloud-host> --port 5556 --com COM3 --id leader_arm_1

--host is how your PC reaches the cloud box's bridge port (see the connectivity notes
in the chat — e.g. an SSH local-forward so --host localhost, or a Brev-exposed host).

Notes:
- The arm must be calibrated under this --id as a LEADER for lerobot. If lerobot
  complains about missing calibration, run its calibrate step for the leader once.
- 'Ctrl-C' to stop; torque is re-enabled / device released on exit.
"""
import argparse
import json
import socket
import time

# lerobot moved the SO-arm leader around between versions: newer installs expose it as
# `so_leader`, the workshop's container uses `so101_leader`. Support both.
try:
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
except ImportError:
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _connect_cloud(host: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


def main():
    ap = argparse.ArgumentParser(description="Stream follower-as-leader joint positions to the cloud sim.")
    ap.add_argument("--host", required=True, help="Cloud box address/host reachable from this PC.")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--com", default="COM3", help="Serial port of the arm (Windows, e.g. COM3).")
    ap.add_argument("--id", default="leader_arm_1", help="lerobot calibration id to load for the arm.")
    ap.add_argument("--fps", type=float, default=30.0, help="Send rate.")
    args = ap.parse_args()

    cfg = SO101LeaderConfig(port=args.com, id=args.id)
    leader = SO101Leader(cfg)
    print(f"[sender] opening arm on {args.com} (id={args.id}) as leader (torque off)...")
    leader.connect()
    print("[sender] arm connected — you can move it by hand now.")

    print(f"[sender] connecting to cloud bridge {args.host}:{args.port} ...")
    sock = None
    while sock is None:
        try:
            sock = _connect_cloud(args.host, args.port)
        except OSError as exc:
            print(f"[sender] cloud not reachable yet ({exc}); retrying in 2s...")
            time.sleep(2)
    print("[sender] connected. Streaming joint positions. Ctrl-C to stop.")

    dt = 1.0 / args.fps
    n = 0
    try:
        while True:
            t0 = time.perf_counter()
            action = leader.get_action()  # {"shoulder_pan.pos": float, ...} in real units
            payload = {k: float(action[k]) for k in JOINTS if k in action}
            line = (json.dumps(payload) + "\n").encode("utf-8")
            try:
                sock.sendall(line)
            except OSError:
                print("[sender] cloud connection dropped; reconnecting...")
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
                while sock is None:
                    try:
                        sock = _connect_cloud(args.host, args.port)
                        print("[sender] reconnected.")
                    except OSError:
                        time.sleep(2)
                continue

            n += 1
            if n % (int(args.fps) * 2 or 1) == 0:
                print(f"[sender] streaming... last gripper={payload.get('gripper.pos'):.1f}")

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\n[sender] stopping...")
    finally:
        try:
            leader.disconnect()
        except Exception:
            pass
        if sock:
            sock.close()
        print("[sender] done.")


if __name__ == "__main__":
    main()

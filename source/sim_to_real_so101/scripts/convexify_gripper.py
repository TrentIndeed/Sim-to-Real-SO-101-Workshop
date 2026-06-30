# SPDX-License-Identifier: Apache-2.0
# The SO-101 gripper + jaw use SDF collision, which is far too slow here (~1 FPS on contact)
# and lets the bottle penetrate. Rewrite those colliders to convexDecomposition (cheap +
# stable, keeps the finger gap) and save a new robot USD next to the original. Run once:
#   python source/sim_to_real_so101/scripts/convexify_gripper.py
# Override APPROX=convexHull (env var) to try a coarser/faster approximation.
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402
try:
    from pxr import PhysxSchema  # noqa: E402
except ImportError:
    PhysxSchema = None

from sim_to_real_so101 import assets  # noqa: E402

USD_DIR = os.path.join(os.path.dirname(os.path.abspath(assets.__file__)), "usd")
IN_USD = os.path.join(USD_DIR, "SO-ARM101-USD.usd")
OUT_USD = os.path.join(USD_DIR, "SO-ARM101-USD-cvx.usd")
TO = os.environ.get("APPROX", "convexDecomposition")


def main():
    print(f"[convexify] opening {IN_USD}")
    stage = Usd.Stage.Open(IN_USD)
    changed = []
    for prim in stage.Traverse():
        low = str(prim.GetPath()).lower()
        if "gripper" not in low and "jaw" not in low:
            continue
        attr = prim.GetAttribute("physics:approximation")
        if not (attr and attr.IsValid()) and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
        if attr and attr.IsValid() and attr.Get():
            old = attr.Get()
            if old != TO:
                attr.Set(TO)
                if PhysxSchema and prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
                    prim.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
                changed.append((str(prim.GetPath()), old, TO))

    for path, old, new in changed:
        print(f"[convexify] {old} -> {new}   {path}")
    print(f"[convexify] total changed: {len(changed)}")

    if changed:
        stage.Export(OUT_USD)
        print(f"[convexify] wrote {OUT_USD}")
    else:
        print("[convexify] NOTHING changed — no sdf colliders found under gripper/jaw. "
              "Paste this output back so we can adjust.")


if __name__ == "__main__":
    main()
    simulation_app.close()

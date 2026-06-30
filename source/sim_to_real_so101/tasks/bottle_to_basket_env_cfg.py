# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# ============================================================================
# Bottle -> Basket task — adapted from vials_to_rack_env_cfg.py for Trenton's
# real setup (pill bottle -> open basket, white foam board). See the dimension
# spec and the joint-replay verification tool under  my_task/ .
#
# STARTING TEMPLATE — it mirrors the vials->rack structure so it loads and runs,
# but a few things are placeholders you must finish (all marked TODO):
#   1. Assets: the bottle currently uses Vial_opaque.usda and the basket uses
#      tray.usda as stand-ins so the env loads. Replace with a real pill-bottle
#      USD (2 in dia x 3 in tall) and a real open-basket USD (10x4x3 in, 1/8 in
#      walls). Easiest = build them as primitives/CAD per my_task/README.md.
#   2. Positions/dims: tune bottle + basket placement to where the arm reaches
#      (use the joint-replay overlay to match your real frames).
#   3. Success box: set the basket inner footprint + rim height for the
#      placement check, and drop the vertical requirement (a bottle in a basket
#      needn't be upright).
# ============================================================================
import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.envs import mdp as base_mdp     # standard Isaac Lab MDP terms

from sim_to_real_so101 import assets
from sim_to_real_so101.assets.so101 import S0101_CONTACT_GRASP_CFG
from sim_to_real_so101.mdp import (
    randomize_sky_light,
    randomize_camera_pose,
    ROBOT_COLORS,
    randomize_mat_rotation,
    randomize_robot_color,
    any_vial_grasped,
    vial_placed_on_rack,
    vial_placed_on_rack_termination,
    time_out,
)

from .so101_env_cfg import EventCfg
from .task_env_cfg import (
    SO101TaskSceneCfg,
    SO101TaskEnvCfg,
    TaskEventCfg,
    TaskObservationsCfg,
    camera_object,            # the shared TiledCameraCfg template (pinhole, opengl convention)
)

assets_path = os.path.dirname(os.path.abspath(assets.__file__))


# ----------------------------------------------------------------------------
# Desk camera placement (look-at).
#
# The workshop bolts the external D455 *inside* a small white photo-tent and only
# jitters it a couple cm. Trenton's real rig is an open foam board with the camera
# in the back-LEFT corner, ~16 in up, looking diagonally across. The exact real
# distance is outside the tent, so we match the ANGLE (left + high, looking across)
# from the tent's left corner instead.
#
# We define our OWN free-standing external camera with an explicit look-at: set
# EYE (where it sits) + TARGET (what it stares at), and the orientation is computed
# exactly. To re-tune, change only EXTERNAL_CAM_EYE — the aim stays locked on the
# workspace, so we never get a blind "all white" from a bad rotation again.
# Coordinates are in the env frame (robot base sits at about (-0.05, 0, 0); the
# bottle spawns near (0.25, 0, 0.05), the basket near (0.18, 0.12, 0.05)).
# +x = forward (where the arm reaches), +z = up.
EXTERNAL_CAM_EYE = (-0.08, 0.31, 0.40)     # back-left, toward the corner, ~16 in up
EXTERNAL_CAM_LOOK = (0.18, 0.06, 0.06)     # workspace centre on the board (bottle<->basket)
EXTERNAL_CAM_YAW_RIGHT_DEG = 15.0          # pan the view to the RIGHT (matches the real rig's "15 deg right")
EXTERNAL_CAM_PITCH_UP_DEG = 5.0            # tilt the view UP (was 15; tilted down 10)


def _look_at_quat_opengl(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (w, x, y, z) that aims an OpenGL-convention camera (looks down
    local -Z, +Y up) from `eye` at `target`. Pure numpy, evaluated at import time."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    z = eye - target                       # opengl +Z points back toward the eye
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:           # eye directly above/below target -> pick another up
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])         # local -> world rotation

    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return (float(w), float(qx), float(qy), float(qz))


def _aim_quat(eye, look, yaw_right_deg=0.0, pitch_up_deg=0.0):
    """Look-at quaternion (opengl) that starts aimed from `eye` at `look`, then pans
    the view RIGHT by `yaw_right_deg` (about world Z) and tilts it UP by `pitch_up_deg`.
    Tune the verbal knobs directly — positive = right / up."""
    eye = np.asarray(eye, dtype=float)
    look = np.asarray(look, dtype=float)
    f = look - eye
    dist = float(np.linalg.norm(f))
    az = np.arctan2(f[1], f[0])                 # azimuth (CCW from +x)
    el = np.arctan2(f[2], np.hypot(f[0], f[1]))  # elevation
    az -= np.radians(yaw_right_deg)             # right = clockwise (looking from above)
    el += np.radians(pitch_up_deg)              # up
    f_new = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    target = eye + f_new * dist
    return _look_at_quat_opengl(eye, target)

# ---- the pick object: a pill bottle (2 in dia x 3 in tall) ----
manipulation_object_base = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ManipulationObject",
    spawn=sim_utils.UsdFileCfg(usd_path=""),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.06)),
)

bottle = manipulation_object_base.replace()
# bottle.usda: a 2 in dia x 3 in tall cylinder body + cap, upright, origin at bottom-centre.
bottle.spawn.usd_path = f"{assets_path}/usd/bottle.usda"
# Physics for the bottle is BAKED INTO bottle.usda (mass + 32 solver iters + depenetration
# + contact offsets). Isaac's runtime modify_* was silently failing on this hand-authored
# USD ("Could not perform modify_mass_properties"), so the props set here were ignored and
# the bottle ran at PhysX defaults (~1 solver iter) -> gripper penetration + contact-storm
# lag. Leaving these None makes Isaac use the USD's baked values as-is.
bottle.spawn.mass_props = None
bottle.spawn.rigid_props = None
bottle.spawn.collision_props = None

# ---- the target: an open basket (10 x 4 x 3 in, 1/8 in walls). basket.usda is a
# floor + 4 walls (open top) at your real dimensions, in basket-local coords. ----
basket = manipulation_object_base.replace()
basket.prim_path = "{ENV_REGEX_NS}/Basket"
basket.spawn.usd_path = f"{assets_path}/usd/basket.usda"
basket.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.15)
basket.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg()
basket.spawn.collision_props = sim_utils.CollisionPropertiesCfg()

BOTTLE_SPAWN_Z = 0.05    # bottom rests on the board; tune to the bottle's half-height


@configclass
class BottleToBasketSceneCfg(SO101TaskSceneCfg):
    # robot with contact sensors enabled (for grasp detection); moved 3 in right (-y)
    robot: ArticulationCfg = S0101_CONTACT_GRASP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=S0101_CONTACT_GRASP_CFG.init_state.replace(pos=(-0.05, -0.076, 0.0)),
    )

    # Trenton's real setup has no black mat — remove it (the lightbox base is the floor).
    mat = None

    # single bottle (vs the workshop's three vials). Kept clear of the basket
    # (basket footprint is ~x[0.31,0.41]); bottle sits closer to the robot.
    bottle = bottle.replace()
    bottle.prim_path = "{ENV_REGEX_NS}/Bottle"
    bottle.init_state.pos = (0.22, 0.0, BOTTLE_SPAWN_Z)
    bottle.init_state.rot = euler_angles_to_quat(np.array([0, 0, 0]), degrees=True)  # upright

    basket = basket.replace()
    basket.prim_path = "{ENV_REGEX_NS}/Basket"
    # Placed to match your real basket spot. (+x = forward, -y = right.)
    basket.init_state.pos = (0.36, 0.07, 0.05)    # spawns just above the surface, settles

    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Bottle"],
    )

    # Replace the workshop's tent-mounted external camera with our own free-standing
    # one, placed by look-at (see EXTERNAL_CAM_EYE/TARGET above). Keeping the scene key
    # "camera_external_D455" means the observation terms (rgb/depth_external_D455) are
    # unchanged — they just read this camera instead. The old tent camera prim is left
    # in place but unused.
    camera_external_D455 = camera_object.replace()
    camera_external_D455.prim_path = "{ENV_REGEX_NS}/external_cam_D455"
    camera_external_D455.offset = TiledCameraCfg.OffsetCfg(
        pos=EXTERNAL_CAM_EYE,
        rot=_aim_quat(EXTERNAL_CAM_EYE, EXTERNAL_CAM_LOOK,
                      EXTERNAL_CAM_YAW_RIGHT_DEG, EXTERNAL_CAM_PITCH_UP_DEG),
        convention="opengl",
    )
    # Sharpen: the template focuses 5 cm ahead (good for the in-tent mount), but our
    # camera is ~0.4 m from the workspace, so refocus there and widen DoF so the whole
    # board is crisp. (Some softness is just the real 640x480 sensor res the policy sees.)
    camera_external_D455.spawn = sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        f_stop=200.0,
        focal_length=13.5,
        focus_distance=0.40,
    )


@configclass
class BottleToBasketDRSceneCfg(BottleToBasketSceneCfg):
    sky_light = AssetBaseCfg(
        prim_path="/World/sky_light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{assets_path}/hdri/moon_lab_1k.exr",
            visible_in_primary_ray=False,
            enable_color_temperature=True,
            color_temperature=6500.0,
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()


@configclass
class BottleToBasketEventCfg(TaskEventCfg):
    """Reset events. Randomize the bottle's start pose on the board; jitter the basket.
    (The workshop's reset_vials_rack is rack-slot specific, so we use the standard
    reset_root_state_uniform instead — ranges are deltas around each asset's init pose.)"""

    # Override the base task's robot color (it hardcodes ["orange"]) — Trenton's arm is white.
    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={"color_names": ["white"]},
    )

    # The mat was removed from the scene, so drop the event that randomized it.
    reset_mat_rotation = None

    # The external camera is now our own free-standing look-at camera (see
    # EXTERNAL_CAM_EYE/TARGET), not the tent mount — so the workshop's mount-jitter
    # event no longer applies. Disable it. (For domain randomization we'll jitter the
    # new camera's pose directly later.)
    reset_camera_external_pose = None

    reset_bottle = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # deltas around bottle.init_state.pos — kept tight so it never lands in/under the basket
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.07, 0.07), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle"),
        },
    )

    reset_basket = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("basket"),
        },
    )


@configclass
class BottleToBasketEventDRCfg(BottleToBasketEventCfg):
    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color, mode="reset",
        params={"color_names": list(ROBOT_COLORS.keys())},
    )
    reset_sky_light = EventTerm(
        func=randomize_sky_light, mode="reset",
        params={"exposure_range": (-4.0, 3.0), "temperature_range": (2500.0, 9500.0),
                "textures_root": f"{assets_path}/hdri", "asset_cfg": SceneEntityCfg("sky_light")},
    )
    reset_mat_rotation = EventTerm(
        func=randomize_mat_rotation, mode="reset",
        params={"yaw_range": (-0.3, 0.3), "asset_cfg": SceneEntityCfg("mat")},
    )


@configclass
class BottleToBasketObservationsCfg(TaskObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        bottle_grasped = ObsTerm(
            func=any_vial_grasped,
            params={"contact_sensor_cfg": SceneEntityCfg("contact_grasp"), "vials": ["bottle"],
                    "min_height": 0.055, "warmup_steps": 30, "force_threshold": 2},
        )
        bottle_placed = ObsTerm(
            func=vial_placed_on_rack,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp"), "vials": ["bottle"],
                "rack_name": "basket", "warmup_steps": 30, "grasp_history_window": 20, "force_threshold": 2,
                # TODO: basket INNER footprint (m) + rim height, in basket-local frame.
                "rack_local_x_min": -0.045, "rack_local_x_max": 0.045,
                "rack_local_y_min": -0.12, "rack_local_y_max": 0.12,
                "rack_local_z_max": 0.0762,          # rim height (3 in)
                "vertical_threshold": 0.0,           # bottle needn't be upright in a basket
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class BottleToBasketTerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    success = DoneTerm(
        func=vial_placed_on_rack_termination, time_out=False,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_grasp"), "vials": ["bottle"],
            "rack_name": "basket", "warmup_steps": 30, "grasp_history_window": 20, "force_threshold": 2,
            "rack_local_x_min": -0.045, "rack_local_x_max": 0.045,
            "rack_local_y_min": -0.12, "rack_local_y_max": 0.12,
            "rack_local_z_max": 0.0762, "vertical_threshold": 0.0,
        },
    )


@configclass
class BottleToBasketEnvCfg(SO101TaskEnvCfg):
    scene: BottleToBasketSceneCfg = BottleToBasketSceneCfg()
    events: BottleToBasketEventCfg = BottleToBasketEventCfg()
    observations: BottleToBasketObservationsCfg = BottleToBasketObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Use the convex-decomposition robot USD (gripper/jaw were SDF -> ~1 FPS on contact).
        # Created by scripts/convexify_gripper.py; falls back to the original if not generated.
        cvx = f"{assets_path}/usd/SO-ARM101-USD-cvx.usd"
        if os.path.exists(cvx):
            self.scene.robot.spawn.usd_path = cvx


@configclass
class BottleToBasketDREnvCfg(BottleToBasketEnvCfg):
    scene: BottleToBasketDRSceneCfg = BottleToBasketDRSceneCfg()
    events: BottleToBasketEventDRCfg = BottleToBasketEventDRCfg()


@configclass
class BottleToBasketEvalEnvCfg(BottleToBasketEnvCfg):
    terminations: BottleToBasketTerminationsCfg = BottleToBasketTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 450 / 60.0


@configclass
class BottleToBasketEvalDREnvCfg(BottleToBasketDREnvCfg):
    terminations: BottleToBasketTerminationsCfg = BottleToBasketTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 450 / 60.0

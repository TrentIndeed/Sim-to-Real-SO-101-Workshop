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
from isaaclab.sensors import ContactSensorCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.envs import mdp as base_mdp     # standard Isaac Lab MDP terms

from sim_to_real_so101 import assets
from sim_to_real_so101.assets.so101 import S0101_CONTACT_GRASP_CFG
from sim_to_real_so101.mdp import (
    randomize_sky_light,
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
)

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

# ---- the pick object: a pill bottle (2 in dia x 3 in tall) ----
manipulation_object_base = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ManipulationObject",
    spawn=sim_utils.UsdFileCfg(usd_path=""),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.06)),
)

bottle = manipulation_object_base.replace()
# bottle.usda: a 2 in dia x 3 in tall cylinder body + cap, upright, origin at bottom-centre.
bottle.spawn.usd_path = f"{assets_path}/usd/bottle.usda"
bottle.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.04)            # ~40 g; set to real
bottle.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(angular_damping=100.0)
bottle.spawn.collision_props = sim_utils.CollisionPropertiesCfg()

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
    # robot with contact sensors enabled (for grasp detection)
    robot: ArticulationCfg = S0101_CONTACT_GRASP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Trenton's real setup has no black mat — remove it (the lightbox base is the floor).
    mat = None

    # single bottle (vs the workshop's three vials)
    bottle = bottle.replace()
    bottle.prim_path = "{ENV_REGEX_NS}/Bottle"
    # TODO: place where the arm comfortably reaches; refine with the replay overlay.
    bottle.init_state.pos = (0.25, 0.0, BOTTLE_SPAWN_Z)
    bottle.init_state.rot = euler_angles_to_quat(np.array([0, 0, 0]), degrees=True)  # upright

    basket = basket.replace()
    basket.prim_path = "{ENV_REGEX_NS}/Basket"
    # TODO: place to match your real basket spot (back-left in your photo).
    basket.init_state.pos = (0.18, 0.12, 0.05)    # spawns just above the surface, settles

    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Bottle"],
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

    reset_bottle = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # TODO: tune to the reachable area; deltas around bottle.init_state.pos
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.10, 0.15), "yaw": (-3.14, 3.14)},
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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp

##
# Pre-defined configs
##

from openarm_pnp.tasks.manager_based.assets.openarm_unimanual import OPEN_ARM_CFG


##
# Scene definition
##


@configclass
class OpenarmPnpSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = OPEN_ARM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "openarm_joint1": 1.57,
                "openarm_joint2": 0.0,
                "openarm_joint3": -1.57,
                "openarm_joint4": 1.57,
                "openarm_joint5": 0.0,
                "openarm_joint6": 0.0,
                "openarm_joint7": 0.0,
                "openarm_finger_joint.*": 0.044,
            },
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.8],
            rot=[1, 0, 0, 0],
        ),
    )

    # Лоток — dynamic, но тяжёлый. Робот может слегка сдвинуть толчком, но не подвинуть.
    tray: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),  # тяжёлый, сопротивляется толчкам
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.55, 0.0, 0.85],
            rot=[1, 0, 0, 0],
        ),
    )

    # Болт: цилиндр r=1.5cm, h=8cm. Спавнится лёжа на боку — гриппер хватает поперёк.
    screw: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Screw",
        spawn=sim_utils.CylinderCfg(
            radius=0.015,
            height=0.08,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.55, 0.6), metallic=0.9, roughness=0.3
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Болт лежит на боку: ось цилиндра по Y. Спавн чуть выше стола (USD pivot внизу),
            # после оседания центр будет ~table_top + radius.
            pos=[0.4, 0.0, 0.86],
            rot=[0.7071, 0.7071, 0.0, 0.0],  # 90° вокруг X → ось цилиндра || Y
        ),
    )

    # end-effector frame sensor
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/EeFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_ee_tcp",
                name="end_effector",
            ),
        ],
    )

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "openarm_joint.*",
        ],
        scale=0.5,
        use_default_offset=True,  # False
    )
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_finger_joint.*"],
        open_command_expr={"openarm_finger_joint.*": 0.044},
        close_command_expr={"openarm_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"]
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"]
                )
            },
        )
        screw_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("screw")},
        )
        tray_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("tray")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Болт рандомизируется около стартовой позиции (как cube в lift task: x±0.1, y±0.25)
    # Базовая позиция [0.4, 0, 0.855] → итог x∈[0.3, 0.5], y∈[-0.15, 0.15]
    # Лоток фиксирован в [0.55, 0, 0.85] → НЕ пересекаются по x
    reset_screw_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.0), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("screw"),
        },
    )

    # Лоток рандомизируется: базовая поз [0.55, 0, 0.85] → x∈[0.45,0.65], y∈[-0.25,0.25]
    # Зона куба x_max=0.4 → 5см запас до лотка
    reset_tray_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tray"),
        },
    )

@configclass
class RewardsCfg:
    # Reaching: EE → cube (tanh, всегда активен)
    reaching_screw = RewTerm(
        func=mdp.reaching_screw,
        weight=1.0,
        params={
            "std": 0.1,
            "screw_cfg": SceneEntityCfg("screw"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )

    # Lifted: бинарный +15/шаг пока куб выше порога (на столе и выше)
    screw_lifted = RewTerm(
        func=mdp.screw_lifted,
        weight=15.0,
        params={"minimal_height": 0.81, "screw_cfg": SceneEntityCfg("screw")},
    )

    # Goal tracking: цель ВНУТРИ лотка (tray_z + 0.05), активна когда куб поднят
    screw_goal_tracking = RewTerm(
        func=mdp.screw_goal_distance,
        weight=16.0,
        params={
            "std": 0.3,
            "minimal_height": 0.81,
            "target_z_offset": 0.05,
            "screw_cfg": SceneEntityCfg("screw"),
            "tray_cfg": SceneEntityCfg("tray"),
        },
    )

    # Success bonus: куб в лотке + почти не движется (vel < 0.1 убивает hover)
    screw_placed = RewTerm(
        func=mdp.screw_placed_in_tray,
        weight=50.0,
        params={
            "horizontal_threshold": 0.08,
            "velocity_threshold": 0.1,
            "screw_cfg": SceneEntityCfg("screw"),
            "tray_cfg": SceneEntityCfg("tray"),
        },
    )

    # Одноразовый большой бонус когда screw_placed_success termination срабатывает.
    # is_terminated_term возвращает 1.0 один раз — на шаге завершения эпизода.
    success_bonus = RewTerm(
        func=mdp.is_terminated_term,
        weight=200.0,
        params={"term_keys": "screw_placed_success"},
    )

    # Штрафы (clamped)
    action_rate = RewTerm(func=mdp.action_rate_l2_clamped, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2_clamped,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"])},
    )


@configclass
class CurriculumCfg:
    """Постепенное усиление штрафов после ~10000 шагов — стабилизация политики."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    screw_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        # Куб в покое z≈0.825, стол top z≈0.85. Threshold 0.78 = ~5см ниже стола
        params={"minimum_height": 0.78, "asset_cfg": SceneEntityCfg("screw")},
    )

    # Успешное завершение: куб в лотке (vel не строгий — termination срабатывает
    # раньше чем куб полностью замер, чтобы не было "выворачивания" во время оседания)
    screw_placed_success = DoneTerm(
        func=mdp.screw_in_tray,
        params={
            "xy_threshold": 0.08,
            "max_height_above_tray": 0.12,
            "max_vel": 0.2,
            "screw_cfg": SceneEntityCfg("screw"),
            "tray_cfg": SceneEntityCfg("tray"),
        },
    )


##
# Environment configuration
##


@configclass
class OpenarmPnpEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: OpenarmPnpSceneCfg = OpenarmPnpSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz, matches lift task
        self.sim.render_interval = self.decimation
        # physx GPU buffers — increase for large num_envs with contact-rich scenes
        self.sim.physx.gpu_max_rigid_patch_count = 2 * 169045  # ~2x max observed in error
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024  # ~2x max observed (16619)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
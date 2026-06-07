from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions, clamped to prevent reward explosion."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    ).clamp(-1000, 1000)


def joint_vel_l2_clamped(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities, clamped to prevent reward explosion."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1).clamp(-1000, 1000)


def reaching_screw(
    env: ManagerBasedRLEnv,
    std: float,
    screw_cfg: SceneEntityCfg = SceneEntityCfg("screw"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for moving end-effector toward screw (tanh-kernel, always active)."""
    screw: RigidObject = env.scene[screw_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    screw_pos = screw.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(screw_pos - ee_pos, dim=1)
    return 1 - torch.tanh(distance / std)


def grasping_screw(
    env: ManagerBasedRLEnv,
    ee_distance_threshold: float,
    screw_cfg: SceneEntityCfg = SceneEntityCfg("screw"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuous reward for closing gripper while near screw.

    distance_factor: 1.0 when EE at screw, fades smoothly (tanh, std=0.05)
    closing_factor: 1.0 when fingers fully closed, 0.0 when fully open
    Product gives gradient signal in BOTH dimensions independently.
    """
    screw: RigidObject = env.scene[screw_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[gripper_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(screw.data.root_pos_w[:, :3] - ee_pos, dim=1)
    distance_factor = 1.0 - torch.tanh(distance / ee_distance_threshold)

    finger_ids, _ = robot.find_joints("openarm_finger_joint.*")
    finger_pos = robot.data.joint_pos[:, finger_ids].mean(dim=1)
    closing_factor = 1.0 - torch.clamp(finger_pos / 0.044, 0.0, 1.0)

    return distance_factor * closing_factor


def screw_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    screw_cfg: SceneEntityCfg = SceneEntityCfg("screw"),
) -> torch.Tensor:
    """Reward = 1 when screw is on table (above minimal_height), 0 when dropped off."""
    screw: RigidObject = env.scene[screw_cfg.name]
    return torch.where(screw.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def screw_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    target_z_offset: float = 0.05,
    screw_cfg: SceneEntityCfg = SceneEntityCfg("screw"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """Награда за приближение куба к точке внутри лотка (tray_pos + z_offset).

    Активна только когда куб поднят (lift task pattern). Цель чуть выше дна лотка,
    что естественно направляет куб ВНУТРЬ, а не сбоку.
    """
    screw: RigidObject = env.scene[screw_cfg.name]
    tray: RigidObject = env.scene[tray_cfg.name]

    target = tray.data.root_pos_w.clone()
    target[:, 2] = target[:, 2] + target_z_offset

    distance = torch.norm(screw.data.root_pos_w - target, dim=1)
    lifted = (screw.data.root_pos_w[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(distance / std))


def screw_placed_in_tray(
    env: ManagerBasedRLEnv,
    horizontal_threshold: float,
    velocity_threshold: float,
    screw_cfg: SceneEntityCfg = SceneEntityCfg("screw"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """Бонус за успех: куб близко к лотку по xy И почти неподвижен.

    velocity_threshold предотвращает 'висит и дрожит' — куб должен реально лежать.
    Без этой проверки агент может получать reward держа куб над лотком.
    """
    screw: RigidObject = env.scene[screw_cfg.name]
    tray: RigidObject = env.scene[tray_cfg.name]

    horizontal_dist = torch.norm(
        screw.data.root_pos_w[:, :2] - tray.data.root_pos_w[:, :2], dim=1
    )
    vel = torch.norm(screw.data.root_lin_vel_w, dim=1)
    return ((horizontal_dist < horizontal_threshold) & (vel < velocity_threshold)).float()

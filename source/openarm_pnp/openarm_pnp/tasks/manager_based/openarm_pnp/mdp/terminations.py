from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def screw_in_tray(
    env: ManagerBasedRLEnv,
    screw_cfg: SceneEntityCfg = SceneEntityCfg("screw"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    xy_threshold: float = 0.08,
    max_height_above_tray: float = 0.06,
    max_vel: float = 0.05,
) -> torch.Tensor:
    """Terminate when screw is placed in tray and at rest.

    Checks per-axis (like pick_place example):
    - xy within threshold of tray center
    - screw above tray but not too high
    - screw velocity below max_vel (object is at rest)
    """
    screw: RigidObject = env.scene[screw_cfg.name]
    tray: RigidObject = env.scene[tray_cfg.name]

    screw_pos = screw.data.root_pos_w
    tray_pos = tray.data.root_pos_w

    xy_dist = torch.norm(screw_pos[:, :2] - tray_pos[:, :2], dim=1)
    screw_z = screw_pos[:, 2]
    tray_z = tray_pos[:, 2]

    screw_vel = torch.norm(screw.data.root_vel_w[:, :3], dim=1)

    in_xy = xy_dist < xy_threshold
    above_tray = screw_z > tray_z
    not_too_high = screw_z < (tray_z + max_height_above_tray)
    at_rest = screw_vel < max_vel

    return in_xy & above_tray & not_too_high & at_rest

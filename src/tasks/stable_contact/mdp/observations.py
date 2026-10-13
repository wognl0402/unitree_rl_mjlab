"""Stable-contact observation terms.

Each term is tagged with how it could be obtained on the real G1:
  - COMMAND-LIKE: supplied by a higher level (operator / perception), the same
    way the velocity command is supplied by the joystick. Actor-safe only if
    that source exists at deployment.
  - PRIVILEGED: simulator-only; use in the critic group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .contact import (
  hand_contact_force_w,
  hand_in_contact,
  normal_force,
  surface_normal_w,
  surface_point_w,
  tangential_component,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "hand_to_target_b",
  "surface_normal_b",
  "hand_contact_state",
  "hand_contact_force",
  "hand_slip_speed",
  "hand_contact_time",
]


def hand_to_target_b(
  env: ManagerBasedRlEnv,
  surface_name: str,
  target_local: tuple[float, float, float],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """[B, 3] target point minus hand site, in the robot base frame.

  COMMAND-LIKE: the hand position is forward kinematics from joint encoders;
  the target must come from perception or the operator.
  """
  robot: Entity = env.scene[asset_cfg.name]
  hand_w = robot.data.site_pos_w[:, asset_cfg.site_ids[0]]
  target_w = surface_point_w(env, surface_name, target_local)
  return quat_apply_inverse(robot.data.root_link_quat_w, target_w - hand_w)


def surface_normal_b(
  env: ManagerBasedRlEnv,
  surface_name: str,
  normal_local: tuple[float, float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """[B, 3] outward surface normal in the robot base frame. COMMAND-LIKE."""
  robot: Entity = env.scene[asset_cfg.name]
  normal_w = surface_normal_w(env, surface_name, normal_local)
  return quat_apply_inverse(robot.data.root_link_quat_w, normal_w)


def hand_contact_state(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """[B, 1] 1.0 if the hand touches the surface. PRIVILEGED (no hand sensor)."""
  return hand_in_contact(env, sensor_name).float().unsqueeze(-1)


def hand_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  surface_name: str,
  normal_local: tuple[float, float, float],
) -> torch.Tensor:
  """[B, 2] (pressing force, tangential force magnitude) in N. PRIVILEGED."""
  force_w = hand_contact_force_w(env, sensor_name)
  normal_w = surface_normal_w(env, surface_name, normal_local)
  f_n = normal_force(force_w, normal_w)
  f_t = torch.linalg.norm(tangential_component(force_w, normal_w), dim=-1)
  return torch.stack([f_n, f_t], dim=-1)


def hand_slip_speed(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  surface_name: str,
  normal_local: tuple[float, float, float],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """[B, 1] in-plane hand speed while in contact (0 otherwise). PRIVILEGED.

  Assumes a static surface, so world velocity equals surface-relative velocity.
  """
  robot: Entity = env.scene[asset_cfg.name]
  vel_w = robot.data.site_lin_vel_w[:, asset_cfg.site_ids[0]]
  normal_w = surface_normal_w(env, surface_name, normal_local)
  speed = torch.linalg.norm(tangential_component(vel_w, normal_w), dim=-1)
  return (speed * hand_in_contact(env, sensor_name).float()).unsqueeze(-1)


def hand_contact_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """[B, 1] seconds of uninterrupted contact. PRIVILEGED.

  Requires track_air_time=True; the sensor clears it on env reset.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_contact_time is not None, (
    f"'{sensor_name}' needs track_air_time=True."
  )
  return sensor.data.current_contact_time[:, :1]

"""Per-step evaluation metrics (MetricsManager reports per-episode averages).

They are chosen to expose fixed-contact failure modes: moving the hand with the
body, pushing harder, repeated breaks, leaning on the wall, freezing, saturating
the active arm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .contact import (
  hand_contact_force_w,
  hand_in_contact,
  normal_force,
  surface_normal_w,
  surface_point_w,
  tangential_component,
)
from .rewards import hand_contact_break, unintended_surface_contact

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "command_speed",
  "base_speed",
  "hand_target_error",
  "hand_contact_ratio",
  "hand_normal_force",
  "hand_force_over_limit",
  "hand_tangential_speed",
  "hand_contact_breaks_per_s",
  "unintended_contact_count",
  "active_arm_limit_margin",
]


def command_speed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Commanded planar body speed (m/s)."""
  return torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)


def base_speed(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Actual planar body speed (m/s). Near zero while commanded = freezing."""
  robot: Entity = env.scene[asset_cfg.name]
  return torch.linalg.norm(robot.data.root_link_lin_vel_b[:, :2], dim=-1)


def hand_target_error(
  env: ManagerBasedRlEnv,
  surface_name: str,
  target_local: tuple[float, float, float],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Hand site to fixed target distance (m). Grows if the hand travels with the body."""
  robot: Entity = env.scene[asset_cfg.name]
  hand_w = robot.data.site_pos_w[:, asset_cfg.site_ids[0]]
  return torch.linalg.norm(surface_point_w(env, surface_name, target_local) - hand_w, dim=-1)


def hand_contact_ratio(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1 while in contact; episode average = contact fraction."""
  return hand_in_contact(env, sensor_name).float()


def hand_normal_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  surface_name: str,
  normal_local: tuple[float, float, float],
) -> torch.Tensor:
  """Pressing force (N). Also logs the batch peak of this step."""
  f_n = normal_force(
    hand_contact_force_w(env, sensor_name), surface_normal_w(env, surface_name, normal_local)
  )
  env.extras.setdefault("log", {})["Metrics/hand_normal_force_peak"] = f_n.max()
  return f_n


def hand_force_over_limit(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  surface_name: str,
  normal_local: tuple[float, float, float],
  force_limit: float,
) -> torch.Tensor:
  """1 while pressing above the penalty limit; average = fraction of time."""
  f_n = normal_force(
    hand_contact_force_w(env, sensor_name), surface_normal_w(env, surface_name, normal_local)
  )
  return (f_n > force_limit).float()


def hand_tangential_speed(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  surface_name: str,
  normal_local: tuple[float, float, float],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """In-plane hand speed while in contact (m/s); should stay ~0 (fixed point)."""
  robot: Entity = env.scene[asset_cfg.name]
  v_t = tangential_component(
    robot.data.site_lin_vel_w[:, asset_cfg.site_ids[0]],
    surface_normal_w(env, surface_name, normal_local),
  )
  return torch.linalg.norm(v_t, dim=-1) * hand_in_contact(env, sensor_name).float()


def hand_contact_breaks_per_s(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Contact-loss events per second (episode average)."""
  return hand_contact_break(env, sensor_name) / env.step_dt


def unintended_contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Non-target robot geoms touching the surface (torso/forearm/other hand...)."""
  return unintended_surface_contact(env, sensor_name)


def active_arm_limit_margin(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Smallest distance of the active-arm joints to their soft limits, as a
  fraction of range. Near 0 = the arm is saturating to hold the point."""
  robot: Entity = env.scene[asset_cfg.name]
  q = robot.data.joint_pos[:, asset_cfg.joint_ids]
  limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  span = (limits[..., 1] - limits[..., 0]).clamp(min=1e-6)
  margin = torch.minimum(q - limits[..., 0], limits[..., 1] - q) / span
  return margin.min(dim=-1).values

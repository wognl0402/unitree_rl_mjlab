"""Stable-contact safety terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .contact import hand_contact_force_w, normal_force, surface_normal_w, surface_point_w

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "hand_contact_force_exceeded",
  "hand_target_error_exceeded",
  "hand_contact_lost",
]


def hand_contact_force_exceeded(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  surface_name: str,
  normal_local: tuple[float, float, float],
  force_limit: float,
) -> torch.Tensor:
  """[B] bool: pressing force above `force_limit` N (safety cutoff)."""
  normal_w = surface_normal_w(env, surface_name, normal_local)
  return normal_force(hand_contact_force_w(env, sensor_name), normal_w) > force_limit


def hand_target_error_exceeded(
  env: ManagerBasedRlEnv,
  surface_name: str,
  target_local: tuple[float, float, float],
  max_error: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """[B] bool: hand site farther than `max_error` m from the fixed target."""
  robot: Entity = env.scene[asset_cfg.name]
  hand_w = robot.data.site_pos_w[:, asset_cfg.site_ids[0]]
  target_w = surface_point_w(env, surface_name, target_local)
  return torch.linalg.norm(target_w - hand_w, dim=-1) > max_error


def hand_contact_lost(
  env: ManagerBasedRlEnv, sensor_name: str, grace_period_s: float
) -> torch.Tensor:
  """[B] bool: no hand contact for longer than `grace_period_s` (includes the
  time since reset if contact was never made). Needs track_air_time."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_air_time is not None
  return sensor.data.current_air_time[:, 0] > grace_period_s

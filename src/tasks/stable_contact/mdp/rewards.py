"""Stable-contact reward terms.

Design rules:
  - No term grows with force: the force reward is a bounded band, and force
    above the band is penalized separately.
  - Slip is measured relative to a desired in-plane hand velocity (zero unless
    a command supplies one), so intentional sliding is not penalized.
  - Thresholds (force band, limits, std) have no defaults; the task must set them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .contact import (
  friction_cone_ratio,
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
  "hand_contact",
  "hand_contact_force_band",
  "hand_contact_force_excess",
  "hand_slip",
  "hand_friction_cone",
  "hand_target_tracking",
  "hand_contact_persistence",
  "hand_contact_break",
  "unintended_surface_contact",
  "body_position_tracking",
]


def hand_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1 while the hand touches the surface."""
  return hand_in_contact(env, sensor_name).float()


def hand_contact_force_band(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  surface_name: str,
  normal_local: tuple[float, float, float],
  force_range: tuple[float, float],
  std: float,
) -> torch.Tensor:
  """1 inside [f_min, f_max]; exp(-(gap/std)^2) outside; 0 without contact."""
  normal_w = surface_normal_w(env, surface_name, normal_local)
  f_n = normal_force(hand_contact_force_w(env, sensor_name), normal_w)
  f_min, f_max = force_range
  gap = torch.clamp(f_min - f_n, min=0.0) + torch.clamp(f_n - f_max, min=0.0)
  reward = torch.exp(-torch.square(gap / std))
  return reward * hand_in_contact(env, sensor_name).float()


def hand_contact_force_excess(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  surface_name: str,
  normal_local: tuple[float, float, float],
  force_limit: float,
) -> torch.Tensor:
  """((f_n - limit) / limit)^2 above the limit, else 0. Use a negative weight."""
  normal_w = surface_normal_w(env, surface_name, normal_local)
  f_n = normal_force(hand_contact_force_w(env, sensor_name), normal_w)
  return torch.square(torch.clamp(f_n - force_limit, min=0.0) / force_limit)


def hand_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  surface_name: str,
  normal_local: tuple[float, float, float],
  asset_cfg: SceneEntityCfg,
  command_name: str | None = None,
) -> torch.Tensor:
  """||v_t - v_t_desired||^2 while in contact. Use a negative weight.

  v_t is the in-plane hand-site velocity (static surface assumed). If
  `command_name` is given, its first 3 entries are a desired world-frame hand
  velocity; only its in-plane part is used. Otherwise the desired velocity is 0.
  """
  robot: Entity = env.scene[asset_cfg.name]
  normal_w = surface_normal_w(env, surface_name, normal_local)
  v_t = tangential_component(robot.data.site_lin_vel_w[:, asset_cfg.site_ids[0]], normal_w)
  if command_name is not None:
    desired = env.command_manager.get_command(command_name)[:, :3]
    v_t = v_t - tangential_component(desired, normal_w)
  cost = torch.sum(torch.square(v_t), dim=-1)
  return cost * hand_in_contact(env, sensor_name).float()


class hand_friction_cone:
  """max(0, |F_t| / (mu F_n) - margin) while pressing. Use a negative weight.

  A ratio near 1 means the contact is at the friction limit (about to slide).
  mu = max(hand geom mu, surface geom mu), MuJoCo's default friction mixing,
  read per env so friction randomization is respected.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    mj_model = env.sim.mj_model
    hand = cfg.params["hand_geom"]
    surface = cfg.params["surface_geom"]
    self._geom_ids = torch.tensor(
      [mj_model.geom(hand).id, mj_model.geom(surface).id], device=env.device
    )

  def _mu(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    friction = env.sim.model.geom_friction
    if friction.dim() == 2:
      friction = friction.unsqueeze(0)
    mu = friction[:, self._geom_ids, 0].max(dim=-1).values
    return mu.expand(env.num_envs)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    surface_name: str,
    normal_local: tuple[float, float, float],
    hand_geom: str,
    surface_geom: str,
    margin: float,
  ) -> torch.Tensor:
    del hand_geom, surface_geom  # Resolved in __init__.
    force_w = hand_contact_force_w(env, sensor_name)
    normal_w = surface_normal_w(env, surface_name, normal_local)
    f_n = normal_force(force_w, normal_w)
    f_t = torch.linalg.norm(tangential_component(force_w, normal_w), dim=-1)
    ratio = friction_cone_ratio(f_n, f_t, self._mu(env))
    return torch.clamp(ratio - margin, min=0.0)


def hand_target_tracking(
  env: ManagerBasedRlEnv,
  surface_name: str,
  target_local: tuple[float, float, float],
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """exp(-||target - hand||^2 / std^2), target given in the surface frame."""
  robot: Entity = env.scene[asset_cfg.name]
  hand_w = robot.data.site_pos_w[:, asset_cfg.site_ids[0]]
  target_w = surface_point_w(env, surface_name, target_local)
  return torch.exp(-torch.sum(torch.square(target_w - hand_w), dim=-1) / std**2)


def hand_contact_persistence(
  env: ManagerBasedRlEnv, sensor_name: str, saturation_time: float
) -> torch.Tensor:
  """min(contact_time, T) / T in [0, 1]. Needs track_air_time.

  The contact timer restarts on any interruption, including breaks that start
  and end between two policy steps, so this captures chatter that
  hand_contact_break cannot see.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_contact_time is not None
  return torch.clamp(sensor.data.current_contact_time[:, 0] / saturation_time, max=1.0)


def hand_contact_break(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1 on the step contact is lost. Use a negative weight. Needs track_air_time.

  Only sees losses still in effect at the end of the policy step; a break that
  starts and ends within one step is missed (see hand_contact_persistence).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.compute_first_air(dt=env.step_dt)[:, 0].float()


def unintended_surface_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Number of non-target robot geoms touching the surface. Use a negative weight."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float().sum(dim=-1)


def body_position_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """exp(-(e / std)^2), e = signed pelvis position error along the displacement
  command's direction (world frame).

  The target is the displacement command's own world-frame goal
  (``target_pos_w = env_origin + displacement * direction``, fixed until the
  command is resampled), not a point that moves with the robot. Motion
  perpendicular to the command axis and vertical motion are not rewarded.
  """
  term = env.command_manager.get_term(command_name)
  robot: Entity = env.scene[asset_cfg.name]
  error = (term.target_pos_w - robot.data.root_link_pos_w[:, :2]) @ term.direction
  return torch.exp(-torch.square(error / std))

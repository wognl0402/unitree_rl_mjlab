"""Core contact quantities shared by observation, reward and termination terms.

Conventions (verified against MuJoCo in scripts/diagnose_hand_contact.py):
  - Hand contact sensors use reduce="netforce", whose `force` is the net force
    the primary (hand) exerts on the secondary (surface), in the world frame.
    In this mode the sensor's normal/tangent/dist fields are not meaningful, so
    the surface normal comes from the surface entity's pose instead.
  - `normal_w` is the surface's outward normal (pointing toward the robot).
  - Pressing force f_n = -(F . n) is >= 0 when the hand pushes into the surface.
  - `found`/`force` are sampled after the step's final forward pass, while the
    sensor's contact/air timers are updated every physics substep from data
    that lags one substep. Near grazing contact the two can disagree by one
    substep (5 ms); use the timers for persistence and break detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "normal_force",
  "tangential_component",
  "friction_cone_ratio",
  "surface_point_w",
  "surface_normal_w",
  "hand_in_contact",
  "hand_contact_force_w",
]


##
# Pure tensor math.
##


def normal_force(force_w: torch.Tensor, normal_w: torch.Tensor) -> torch.Tensor:
  """Pressing force [B] from force exerted on the surface [B, 3] and outward normal."""
  return -torch.sum(force_w * normal_w, dim=-1)


def tangential_component(vec_w: torch.Tensor, normal_w: torch.Tensor) -> torch.Tensor:
  """Component of vec_w [B, 3] lying in the surface plane."""
  return vec_w - torch.sum(vec_w * normal_w, dim=-1, keepdim=True) * normal_w


def friction_cone_ratio(
  f_normal: torch.Tensor, f_tangential: torch.Tensor, mu: torch.Tensor | float,
  eps: float = 1e-6,
) -> torch.Tensor:
  """|F_t| / (mu * F_n). Values near 1 mean the contact is at the slip limit.

  Returns 0 when there is no pressing force.
  """
  pressing = f_normal > eps
  ratio = f_tangential / (mu * f_normal.clamp(min=eps))
  return torch.where(pressing, ratio, torch.zeros_like(ratio))


##
# Environment accessors.
##


def surface_normal_w(
  env: ManagerBasedRlEnv,
  surface_name: str,
  normal_local: tuple[float, float, float],
) -> torch.Tensor:
  """Outward surface normal [B, 3] in world frame from the surface entity pose."""
  quat_w = env.scene[surface_name].data.root_link_quat_w
  local = torch.tensor(normal_local, device=quat_w.device, dtype=quat_w.dtype)
  return quat_apply(quat_w, local.expand(quat_w.shape[0], 3))


def surface_point_w(
  env: ManagerBasedRlEnv,
  surface_name: str,
  point_local: tuple[float, float, float],
) -> torch.Tensor:
  """A point [B, 3] given in the surface body frame, expressed in world frame."""
  data = env.scene[surface_name].data
  quat_w = data.root_link_quat_w
  local = torch.tensor(point_local, device=quat_w.device, dtype=quat_w.dtype)
  return data.root_link_pos_w + quat_apply(quat_w, local.expand(quat_w.shape[0], 3))


def hand_in_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """[B] bool: the sensor's primary (hand) touches its secondary (surface)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None, f"'{sensor_name}' must record 'found'."
  return sensor.data.found[:, 0] > 0


def hand_contact_force_w(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """[B, 3] net force the hand exerts on the surface (world frame)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.cfg.reduce == "netforce", (
    f"'{sensor_name}' must use reduce='netforce' for a world-frame net force."
  )
  assert sensor.data.force is not None, f"'{sensor_name}' must record 'force'."
  return sensor.data.force[:, 0]

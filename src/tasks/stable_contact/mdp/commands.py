"""Bounded body-displacement command, delivered as a velocity command.

A target base position is sampled at a bounded offset from the env origin
along a fixed direction (parallel to the contact surface). Each step it is
converted into a body-frame twist command

    v_xy = clip_norm(position_gain * (target - base_xy), max_speed)
    w_z  = clip(heading_gain * wrap(heading - yaw), +-max_yaw_rate)

so the command has the same (vx, vy, wz) layout as the velocity task and all
velocity-task observation/reward terms apply unchanged. Unlike a sampled
velocity, the body displacement is bounded by `displacement_range`, which keeps
a fixed hand contact point reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["BaseDisplacementCommand", "BaseDisplacementCommandCfg"]


class BaseDisplacementCommand(CommandTerm):
  cfg: BaseDisplacementCommandCfg

  def __init__(self, cfg: BaseDisplacementCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    direction = torch.tensor(cfg.direction, device=self.device, dtype=torch.float32)
    self.direction = direction / torch.linalg.norm(direction)
    self.displacement = torch.zeros(self.num_envs, device=self.device)
    self.target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    pos_err = torch.linalg.norm(self.target_pos_w - self.robot.data.root_link_pos_w[:, :2], dim=-1)
    vel_err = torch.linalg.norm(
      self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
    )
    self.metrics["error_pos"] += pos_err / steps
    self.metrics["error_vel_xy"] += vel_err / steps

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    d = r.uniform_(-self.cfg.displacement_range, self.cfg.displacement_range)
    standing = torch.empty_like(d).uniform_(0.0, 1.0) < self.cfg.rel_standing_envs
    self.displacement[env_ids] = torch.where(standing, torch.zeros_like(d), d)
    origin = self._env.scene.env_origins[env_ids, :2]
    self.target_pos_w[env_ids] = origin + self.displacement[env_ids, None] * self.direction

  def _update_command(self) -> None:
    data = self.robot.data
    error_w = self.target_pos_w - data.root_link_pos_w[:, :2]
    vel_w = self.cfg.position_gain * error_w
    speed = torch.linalg.norm(vel_w, dim=-1, keepdim=True)
    vel_w = vel_w * torch.clamp(self.cfg.max_speed / speed.clamp(min=1e-6), max=1.0)
    yaw = data.heading_w
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    self.vel_command_b[:, 0] = cos * vel_w[:, 0] + sin * vel_w[:, 1]
    self.vel_command_b[:, 1] = -sin * vel_w[:, 0] + cos * vel_w[:, 1]
    heading_error = wrap_to_pi(self.cfg.heading - yaw)
    self.vel_command_b[:, 2] = torch.clamp(
      self.cfg.heading_gain * heading_error, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate
    )


@dataclass(kw_only=True)
class BaseDisplacementCommandCfg(CommandTermCfg):
  entity_name: str
  direction: tuple[float, float]
  """Motion direction in the world/env frame (normalized internally)."""
  displacement_range: float
  """Target offset from the env origin is sampled in [-range, range] (m)."""
  position_gain: float
  """1/s: commanded speed per meter of position error."""
  max_speed: float
  """m/s cap on the commanded planar speed."""
  heading: float = 0.0
  """World yaw the body should keep (rad)."""
  heading_gain: float = 0.5
  max_yaw_rate: float = 0.5
  rel_standing_envs: float = 0.0
  """Fraction of resamples whose target is the env origin (hold still)."""

  def build(self, env: ManagerBasedRlEnv) -> BaseDisplacementCommand:
    return BaseDisplacementCommand(self, env)

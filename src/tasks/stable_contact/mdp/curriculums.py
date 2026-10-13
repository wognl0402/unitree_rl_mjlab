"""Step-based curriculum for the displacement command (no adaptive logic)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import BaseDisplacementCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["displacement_range_stages"]


def displacement_range_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: tuple[tuple[int, float], ...],
) -> torch.Tensor:
  """Set the displacement range from (env_step, range) stages, latest reached wins."""
  del env_ids  # Unused.
  cfg = env.command_manager.get_term(command_name).cfg
  assert isinstance(cfg, BaseDisplacementCommandCfg)
  for step, value in stages:
    if env.common_step_counter >= step:
      cfg.displacement_range = value
  return torch.tensor([cfg.displacement_range])

"""EXPERIMENTAL actor whose translational command input is amplified (actor only).

The actor sees ``clip_norm(gain * (vx, vy), max_speed)`` in place of the real
(vx, vy) twist command. The transform is applied to the raw observation just
before the empirical normalizer's forward pass, while the normalizer is still
updated with the *real* observation (``update_normalization`` is inherited).
Its statistics therefore evolve exactly as for an unmodified actor and the gain
cannot be absorbed during training. The environment, real command, rewards,
reward/phase gates and critic are untouched; wz and every other dimension pass
through unchanged.
"""

from __future__ import annotations

import torch
from rsl_rl.models.mlp_model import MLPModel
from torch import nn

# Actor observation layout (StableContactMove): base_ang_vel 3, projected_gravity 3, command 3.
COMMAND_XY = (6, 8)
MAX_SPEED = 0.2  # BaseDisplacementCommandCfg.max_speed for the movement tasks (m/s)


def amplify_command(x: torch.Tensor, gain: float, start: int = COMMAND_XY[0], stop: int = COMMAND_XY[1],
                    max_speed: float = MAX_SPEED) -> torch.Tensor:
  """Return a copy of x with x[..., start:stop] scaled by gain and norm-clipped to max_speed."""
  x = x.clone()
  v = x[..., start:stop] * gain
  n = torch.linalg.norm(v, dim=-1, keepdim=True)
  x[..., start:stop] = v * torch.clamp(max_speed / n.clamp(min=1e-9), max=1.0)
  return x


class _ExportCommandGain(nn.Module):
  """Prepends the command transform to an exported (ONNX/JIT) policy module."""

  def __init__(self, base: nn.Module, gain: float) -> None:
    super().__init__()
    self.base = base
    self.gain = gain

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.base(amplify_command(x, self.gain))

  def __getattr__(self, name):
    try:
      return super().__getattr__(name)
    except AttributeError:
      return getattr(super().__getattr__("base"), name)


class ActorCommandGainMLPModel(MLPModel):
  """MLPModel with a fixed actor-only translational command gain (set by subclasses)."""

  command_gain: float = 1.0

  def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
    latent = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
    return self.obs_normalizer(amplify_command(latent, self.command_gain))

  def as_jit(self) -> nn.Module:
    return _ExportCommandGain(super().as_jit(), self.command_gain)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _ExportCommandGain(super().as_onnx(verbose), self.command_gain)


class ActorCommandGain1p5(ActorCommandGainMLPModel):
  command_gain = 1.5

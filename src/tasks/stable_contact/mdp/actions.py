"""Joint-position action whose zero point holds the default pose against gravity.

The G1 uses MuJoCo position servos (force = kp * (ctrl - q) - kd * qdot, no
gravity compensation). With the stock action offset (ctrl = default_joint_pos
at zero action) a joint carrying a gravity load g settles at q = default - g/kp.
For the contact-ready active arm that sag (~0.2 rad at the shoulder) pulls the
palm off the wall within ~65 ms of reset.

GravityHoldJointPositionAction adds g(q_default) / kp to the offset of the
selected joints, so zero action is a static equilibrium of those joints at the
default pose. g is MuJoCo's qfrc_bias at zero velocity (gravity only, no
contact) evaluated on the env's compiled model, which is exact for joints whose
subtree carries no external load (arm, waist). Leg joints share load through
the feet and are not a valid target for this per-joint correction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "GravityHoldJointPositionActionCfg",
  "GravityHoldJointPositionAction",
  "gravity_hold_offsets",
]


def gravity_hold_offsets(
  m: mujoco.MjModel,
  entity_name: str,
  root_quat: np.ndarray,
  root_height: float,
  joint_pos: dict[str, float],
  hold_joints: list[str],
) -> np.ndarray:
  """Position-target offsets (rad) that make each `hold_joints` servo cancel gravity
  at the given pose. Raises if an actuator is not a plain position servo or the
  required torque exceeds its force limit."""
  d = mujoco.MjData(m)
  free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
          and m.joint(j).name.startswith(f"{entity_name}/")]
  if len(free) != 1:
    raise ValueError(f"Expected one free joint for {entity_name}, found {len(free)}")
  a = m.jnt_qposadr[free[0]]
  d.qpos[a:a + 3] = (0.0, 0.0, root_height)
  d.qpos[a + 3:a + 7] = root_quat
  for name, value in joint_pos.items():
    d.qpos[m.jnt_qposadr[m.joint(f"{entity_name}/{name}").id]] = value
  mujoco.mj_forward(m, d)  # qvel = 0 -> qfrc_bias is the gravity term only.
  out = np.zeros(len(hold_joints))
  for i, name in enumerate(hold_joints):
    full = f"{entity_name}/{name}"
    act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
    if act < 0 or m.actuator_trnid[act, 0] != m.joint(full).id:
      raise ValueError(f"No joint actuator for {full}")
    kp, b = m.actuator_gainprm[act, 0], m.actuator_biasprm[act]
    if m.actuator_gear[act, 0] != 1.0 or not np.isclose(b[1], -kp) or m.actuator_dyntype[act] != 0:
      raise ValueError(f"{full} is not a unit-gear position servo (gain {kp}, bias {b[:3]})")
    tau = float(d.qfrc_bias[m.jnt_dofadr[m.joint(full).id]])
    if m.actuator_forcelimited[act] and not (m.actuator_forcerange[act, 0] < tau < m.actuator_forcerange[act, 1]):
      raise ValueError(f"{full}: gravity torque {tau:.2f} N m exceeds force range")
    out[i] = tau / kp
  return out


@dataclass(kw_only=True)
class GravityHoldJointPositionActionCfg(JointPositionActionCfg):
  hold_joint_names: tuple[str, ...] = ()
  """Regexes of joints whose zero-action target is gravity-compensated."""

  def build(self, env: ManagerBasedRlEnv) -> GravityHoldJointPositionAction:
    return GravityHoldJointPositionAction(self, env)


class GravityHoldJointPositionAction(JointPositionAction):
  def __init__(self, cfg: GravityHoldJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    assert cfg.use_default_offset, "gravity hold is defined relative to the default pose"
    cols = [i for i, n in enumerate(self._target_names)
            if any(re.fullmatch(p, n) for p in cfg.hold_joint_names)]
    if not cols:
      self.gravity_offset = torch.zeros_like(self._offset[0])
      return
    data = self._entity.data
    names = self._entity.joint_names
    joint_pos = dict(zip(names, data.default_joint_pos[0].tolist()))
    root = data.default_root_state[0]
    off = gravity_hold_offsets(
      env.sim.mj_model, cfg.entity_name, root[3:7].cpu().numpy(), float(root[2]),
      joint_pos, [self._target_names[i] for i in cols])
    self.gravity_offset = torch.zeros_like(self._offset[0])
    self.gravity_offset[cols] = torch.tensor(off, dtype=self._offset.dtype, device=self._offset.device)
    self._offset = self._offset + self.gravity_offset

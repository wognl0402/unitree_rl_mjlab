"""Contact-ready initial pose for the G1 (forward kinematics on g1.xml, no physics).

Solves the active arm's 7 joints so the palm site reaches a world point with
the robot at its spawn pose and every other joint at HOME_KEYFRAME. A null-space
pull toward mid-range keeps the solution away from joint limits.
"""

import functools
import re

import mujoco
import numpy as np

from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME, get_spec

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")


@functools.lru_cache(maxsize=1)
def _model() -> mujoco.MjModel:
  return get_spec().compile()


def home_joint_positions() -> dict[str, float]:
  """HOME_KEYFRAME expanded to explicit joint names (unmatched joints -> 0)."""
  m = _model()
  out = {}
  for j in range(m.njnt):
    name = m.joint(j).name
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    matches = [v for pat, v in HOME_KEYFRAME.joint_pos.items() if re.fullmatch(pat, name)]
    out[name] = float(matches[0]) if matches else 0.0
  return out


def _set_spawn(
  data: mujoco.MjData, m: mujoco.MjModel, joints: dict[str, float], height: float | None = None
) -> None:
  data.qpos[:] = 0.0
  data.qpos[:3] = HOME_KEYFRAME.pos
  if height is not None:
    data.qpos[2] = height
  data.qpos[3:7] = HOME_KEYFRAME.rot
  for name, value in joints.items():
    data.qpos[m.jnt_qposadr[m.joint(name).id]] = value


@functools.lru_cache(maxsize=1)
def standing_pelvis_height() -> float:
  """Pelvis height at which the HOME-pose foot collision capsules touch z = 0.

  HOME_KEYFRAME spawns the pelvis higher, so the robot drops ~1.6 cm on reset.
  """
  m = _model()
  d = mujoco.MjData(m)
  _set_spawn(d, m, home_joint_positions())
  mujoco.mj_kinematics(m, d)
  lowest = np.inf
  for g in range(m.ngeom):
    if not re.fullmatch(r"(left|right)_foot\d+_collision", m.geom(g).name):
      continue
    radius = m.geom_size[g][0]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
      axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1]
      z = min(d.geom_xpos[g][2] + axis[2], d.geom_xpos[g][2] - axis[2])
    elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
      z = d.geom_xpos[g][2]
    else:
      raise ValueError(f"Unsupported foot geom type for {m.geom(g).name}")
    lowest = min(lowest, z - radius)
  return float(HOME_KEYFRAME.pos[2] - lowest)


def palm_position(hand: str, joints: dict[str, float], height: float | None = None) -> np.ndarray:
  """Palm site position (world) at the spawn pose with the given joint values."""
  m = _model()
  d = mujoco.MjData(m)
  _set_spawn(d, m, joints, height)
  mujoco.mj_kinematics(m, d)
  return d.site(f"{hand}_palm").xpos.copy()


def hand_capsule_depth(
  m: mujoco.MjModel, d: mujoco.MjData, hand: str, face_point: np.ndarray, inward: np.ndarray
) -> float:
  """Deepest penetration (m, >0 = inside) of the hand collision capsule past a
  plane through `face_point` with inward normal `inward`."""
  g = m.geom(f"{hand}_hand_collision").id
  half_axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1]
  ends = (d.geom_xpos[g] + half_axis, d.geom_xpos[g] - half_axis)
  return max(float(np.dot(e - face_point, inward)) for e in ends) + m.geom_size[g][0]


@functools.lru_cache(maxsize=8)
def contact_ready_arm_pose(
  hand: str,
  palm_target: tuple[float, float, float],
  pelvis_height: float,
  wall_inward_normal: tuple[float, float, float],
  soft_limit_factor: float = 0.9,
) -> tuple[tuple[str, float], ...]:
  """Active-arm joint positions placing the palm at `palm_target`, with the
  hand capsule parallel to the wall (flat palm), the pelvis at `pelvis_height`
  (env frame) and every other joint at HOME_KEYFRAME.

  Without the orientation constraint the 9 cm hand capsule can end up partly
  inside the wall even though the palm site is on the surface.
  Raises if the pose is not reachable within the soft joint limits.
  """
  m = _model()
  d = mujoco.MjData(m)
  _set_spawn(d, m, home_joint_positions(), pelvis_height)
  names = [f"{hand}_{j}_joint" for j in ARM_JOINTS]
  jids = [m.joint(n).id for n in names]
  qadr = np.array([m.jnt_qposadr[j] for j in jids])
  dofs = [m.jnt_dofadr[j] for j in jids]
  lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
  center, half = (lo + hi) / 2, (hi - lo) / 2
  soft_lo, soft_hi = center - soft_limit_factor * half, center + soft_limit_factor * half
  site = m.site(f"{hand}_palm").id
  geom = m.geom(f"{hand}_hand_collision").id
  body = m.geom_bodyid[geom]
  normal = np.asarray(wall_inward_normal, dtype=float)
  target = np.asarray(palm_target, dtype=float)
  for _ in range(1000):
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    axis = d.geom_xmat[geom].reshape(3, 3)[:, 2]
    err = np.append(target - d.site_xpos[site], -np.dot(axis, normal))
    if np.linalg.norm(err) < 1e-5:
      break
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, None, site)
    mujoco.mj_jacBody(m, d, None, jacr, body)
    # d(axis . n)/dt = (w x axis) . n = w . (axis x n)
    J = np.vstack([jacp[:, dofs], np.cross(axis, normal) @ jacr[:, dofs]])
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(4), err)
    null = (np.eye(len(dofs)) - np.linalg.pinv(J) @ J) @ (0.05 * (center - d.qpos[qadr]))
    d.qpos[qadr] = np.clip(d.qpos[qadr] + 0.5 * dq + null, soft_lo, soft_hi)
  mujoco.mj_kinematics(m, d)
  residual = float(np.linalg.norm(target - d.site_xpos[site]))
  tilt = abs(float(np.dot(d.geom_xmat[geom].reshape(3, 3)[:, 2], normal)))
  if residual > 1e-3 or tilt > 1e-2:
    raise ValueError(
      f"{hand} palm cannot reach {palm_target} flat: residual {residual:.4f} m, tilt {tilt:.3f}"
    )
  return tuple((n, float(d.qpos[a])) for n, a in zip(names, qadr))

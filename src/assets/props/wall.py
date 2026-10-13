"""Fixed vertical wall used as a contact surface.

Frame convention (wall body frame):
  - The body origin is the center of the contact face.
  - The contact face is the plane x = 0; the box extends into +x.
  - The outward surface normal (pointing toward the robot side) is -x.

Because the entity has no joints, mjlab wraps it as a mocap body. It is only
placed per environment by a reset event (mdp.reset_root_state_uniform), which
adds env_origins to the init_state pose.
"""

import math

import mujoco

from mjlab.entity import EntityCfg

WALL_GEOM_NAME = "wall_geom"
WALL_OUTWARD_NORMAL_LOCAL = (-1.0, 0.0, 0.0)


def get_wall_spec(
  width: float = 2.0,
  height: float = 2.0,
  thickness: float = 0.05,
  friction: tuple[float, float, float] = (1.0, 0.005, 0.0001),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="wall")
  body.add_geom(
    name=WALL_GEOM_NAME,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(thickness / 2, width / 2, height / 2),
    pos=(thickness / 2, 0.0, 0.0),
    friction=friction,
    condim=3,
    rgba=(0.6, 0.6, 0.7, 0.5),
  )
  return spec


def wall_init_state(
  yaw: float, distance: float, face_center_height: float
) -> EntityCfg.InitialStateCfg:
  """Place the face `distance` m from the env origin along heading `yaw`.

  The wall's local +x (into the wall) is rotated to point along `yaw`, so the
  outward normal points back toward the env origin.
  """
  return EntityCfg.InitialStateCfg(
    pos=(distance * math.cos(yaw), distance * math.sin(yaw), face_center_height),
    rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
  )


def get_wall_cfg(
  init_state: EntityCfg.InitialStateCfg,
  width: float = 2.0,
  height: float = 2.0,
  thickness: float = 0.05,
  friction: tuple[float, float, float] = (1.0, 0.005, 0.0001),
) -> EntityCfg:
  return EntityCfg(
    init_state=init_state,
    spec_fn=lambda: get_wall_spec(width, height, thickness, friction),
  )

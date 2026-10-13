"""StableContactStand: DEVELOPMENT / DIAGNOSTIC harness, NOT final controller behavior.

Composition:
    existing Unitree-G1-Flat config (29-DoF JointPositionAction, PD actuators,
    proprioceptive actor obs, stability/safety rewards)
  + motion objective "stand": the existing twist command held at zero
  + a fixed wall beside the active hand
  + hand/wall contact sensors and contact observations
  + contact rewards, only when the caller supplies weights and thresholds

Deliberately not registered as a task: reward weights and thresholds are open
research decisions. Geometry defaults come from a forward-kinematics reach
study of g1.xml (palm reach at chest height: 0.59 m lateral, 0.46 m forward;
palm at the home pose: |y| = 0.23 m).
"""

import math
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.spec_config import CollisionCfg

import src.tasks.stable_contact.mdp as mdp
from src.assets.props import get_wall_cfg, wall_init_state
from src.assets.props.wall import WALL_GEOM_NAME, WALL_OUTWARD_NORMAL_LOCAL
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

SURFACE = "wall"
# Hand capsule radius in g1.xml; the palm site sits on the capsule axis.
HAND_CAPSULE_RADIUS = 0.035


@dataclass(frozen=True)
class G1Hand:
  site: str
  geom: str
  arm_joints_regex: str
  side_yaw: float  # Heading from the robot to the wall on this hand's side.
  forward_sign: float  # Wall-frame y that points along robot +x.


G1_HANDS = {
  "left": G1Hand("left_palm", "left_hand_collision", r"left_(shoulder|elbow|wrist)_.*",
                 math.pi / 2, -1.0),
  "right": G1Hand("right_palm", "right_hand_collision", r"right_(shoulder|elbow|wrist)_.*",
                  -math.pi / 2, 1.0),
}


def hand_sensor_name(hand: str) -> str:
  return f"{hand}_hand_{SURFACE}_contact"


UNINTENDED_SENSOR = f"robot_{SURFACE}_unintended_contact"

# Same as g1_constants.FULL_COLLISION, except the hand capsules get condim=3 so
# hand/wall contacts carry friction (condim=1 is frictionless).
G1_COLLISION_WITH_HAND_FRICTION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    r"^(left|right)_hand_collision$": 3,
    ".*_collision": 1,
  },
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)


def _hand_site_cfg(hand: str) -> SceneEntityCfg:
  # A fresh instance per term: managers resolve and validate it in place.
  return SceneEntityCfg("robot", site_names=(G1_HANDS[hand].site,))


def contact_target_local(
  hand: str, forward: float, height_offset: float
) -> tuple[float, float, float]:
  """Palm-site target in the wall frame: capsule radius off the face."""
  return (-HAND_CAPSULE_RADIUS, G1_HANDS[hand].forward_sign * forward, height_offset)


def make_contact_reward_terms(
  active_hand: str,
  weights: dict[str, float],
  *,
  force_range: tuple[float, float],
  force_std: float,
  force_limit: float,
  target_local: tuple[float, float, float],
  target_std: float,
  cone_margin: float,
  persistence_time: float,
) -> dict[str, RewardTermCfg]:
  """Build the stable-contact reward terms named in `weights`.

  All thresholds are required: their values are a research decision.
  """
  hand = G1_HANDS[active_hand]
  sensor = hand_sensor_name(active_hand)
  surface = {"surface_name": SURFACE, "normal_local": WALL_OUTWARD_NORMAL_LOCAL}
  available = {
    "hand_contact": (mdp.hand_contact, {"sensor_name": sensor}),
    "hand_force_band": (mdp.hand_contact_force_band, {
      "sensor_name": sensor, **surface, "force_range": force_range, "std": force_std}),
    "hand_force_excess": (mdp.hand_contact_force_excess, {
      "sensor_name": sensor, **surface, "force_limit": force_limit}),
    "hand_slip": (mdp.hand_slip, {
      "sensor_name": sensor, **surface, "asset_cfg": _hand_site_cfg(active_hand)}),
    "hand_friction_cone": (mdp.hand_friction_cone, {
      "sensor_name": sensor, **surface, "hand_geom": f"robot/{hand.geom}",
      "surface_geom": f"{SURFACE}/{WALL_GEOM_NAME}", "margin": cone_margin}),
    "hand_target": (mdp.hand_target_tracking, {
      "surface_name": SURFACE, "target_local": target_local, "std": target_std,
      "asset_cfg": _hand_site_cfg(active_hand)}),
    "hand_contact_persistence": (mdp.hand_contact_persistence, {
      "sensor_name": sensor, "saturation_time": persistence_time}),
    "hand_contact_break": (mdp.hand_contact_break, {"sensor_name": sensor}),
    "unintended_contact": (mdp.unintended_surface_contact, {"sensor_name": UNINTENDED_SENSOR}),
  }
  unknown = set(weights) - set(available)
  if unknown:
    raise ValueError(f"Unknown contact reward terms: {sorted(unknown)}")
  return {
    name: RewardTermCfg(func=available[name][0], weight=w, params=available[name][1])
    for name, w in weights.items()
  }


def make_stable_contact_stand_env_cfg(
  active_hand: str = "left",
  wall_distance: float = 0.40,
  wall_face_height: float = 1.0,
  target_forward: float = 0.15,
  target_height_offset: float = 0.0,
  contact_rewards: dict[str, RewardTermCfg] | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """DEVELOPMENT harness. See module docstring."""
  if active_hand not in G1_HANDS:
    raise ValueError(f"active_hand must be one of {list(G1_HANDS)}, got {active_hand!r}")
  hand = G1_HANDS[active_hand]
  cfg = unitree_g1_flat_env_cfg(play=play)

  # Scene: robot (hand friction enabled) + wall on the active hand's side.
  cfg.scene.entities["robot"].collisions = (G1_COLLISION_WITH_HAND_FRICTION,)
  cfg.scene.entities[SURFACE] = get_wall_cfg(
    wall_init_state(hand.side_yaw, wall_distance, wall_face_height)
  )
  hand_sensors = tuple(
    ContactSensorCfg(
      name=hand_sensor_name(side),
      primary=ContactMatch(mode="geom", pattern=G1_HANDS[side].geom, entity="robot"),
      secondary=ContactMatch(mode="geom", pattern=WALL_GEOM_NAME, entity=SURFACE),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    )
    for side in G1_HANDS
  )
  unintended = ContactSensorCfg(
    name=UNINTENDED_SENSOR,
    primary=ContactMatch(mode="geom", pattern=r".*_collision", entity="robot",
                         exclude=(hand.geom,)),
    secondary=ContactMatch(mode="geom", pattern=WALL_GEOM_NAME, entity=SURFACE),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + hand_sensors + (unintended,)

  # Events: the wall is placed relative to the env origin, so the robot spawns
  # there facing +x (the velocity task's +-pi yaw randomization is removed).
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.events["reset_wall"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={"pose_range": {}, "velocity_range": {},
            "asset_cfg": SceneEntityCfg(SURFACE)},
  )

  # Motion objective "stand": every env receives a zero twist command.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.rel_standing_envs = 1.0
  cfg.curriculum.pop("command_vel", None)

  # Posture terms must not pull the active arm back to the default pose.
  not_active_arm = (rf"^(?!{hand.arm_joints_regex}$).*",)
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=not_active_arm)
  cfg.rewards["stand_still"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=not_active_arm
  )

  # Observations.
  target_local = contact_target_local(active_hand, target_forward, target_height_offset)
  surface = {"surface_name": SURFACE, "normal_local": WALL_OUTWARD_NORMAL_LOCAL}
  sensor = hand_sensor_name(active_hand)

  def command_like() -> dict[str, ObservationTermCfg]:
    return {
      "hand_to_target": ObservationTermCfg(
        func=mdp.hand_to_target_b,
        params={"surface_name": SURFACE, "target_local": target_local,
                "asset_cfg": _hand_site_cfg(active_hand)},
      ),
      "surface_normal": ObservationTermCfg(func=mdp.surface_normal_b, params=dict(surface)),
    }

  privileged = {
    "hand_contact": ObservationTermCfg(
      func=mdp.hand_contact_state, params={"sensor_name": sensor}),
    "hand_force": ObservationTermCfg(
      func=mdp.hand_contact_force, params={"sensor_name": sensor, **surface}),
    "hand_slip": ObservationTermCfg(
      func=mdp.hand_slip_speed,
      params={"sensor_name": sensor, **surface, "asset_cfg": _hand_site_cfg(active_hand)}),
    "hand_contact_time": ObservationTermCfg(
      func=mdp.hand_contact_time, params={"sensor_name": sensor}),
  }
  cfg.observations["actor"].terms.update(command_like())
  cfg.observations["critic"].terms.update({**command_like(), **privileged})

  if contact_rewards:
    cfg.rewards.update(contact_rewards)

  cfg.viewer.body_name = "torso_link"
  return cfg

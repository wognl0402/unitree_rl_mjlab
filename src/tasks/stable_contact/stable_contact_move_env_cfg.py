"""StableContactMove: EXPERIMENTAL fixed-point contact baseline.

Task: move the body (bounded displacement parallel to the wall) while one palm
stays on a fixed world-space point on the wall, and stay upright.

Built on the StableContactStand harness (wall, hand sensors, contact obs), which
is built on Unitree-G1-Flat (29-DoF JointPositionAction, PD actuators). Every
tunable value lives in StableContactMoveCfg.

Training simplifications for V1 (documented, not research claims):
  - Episodes start with the palm already on the target (light preload), so the
    policy does not have to find or reach the wall.
  - The active arm's default pose is the contact-ready pose, so zero action
    holds the hand on the target. This changes the 7 active-arm entries of
    default_joint_pos (= JointPositionAction offset); joint order, action
    scales and PD gains are unchanged.
  - External pushes are disabled.
"""

from dataclasses import dataclass, field, fields

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

import src.tasks.stable_contact.mdp as mdp
from src.assets.props.wall import WALL_OUTWARD_NORMAL_LOCAL
from src.assets.robots.unitree_g1.g1_constants import HOME_KEYFRAME
from src.tasks.stable_contact.contact_pose import (
  contact_ready_arm_pose,
  home_joint_positions,
  standing_pelvis_height,
)
from src.tasks.stable_contact.stable_contact_env_cfg import (
  G1_HANDS,
  HAND_CAPSULE_RADIUS,
  SURFACE,
  UNINTENDED_SENSOR,
  contact_target_local,
  hand_sensor_name,
  make_contact_reward_terms,
  make_stable_contact_stand_env_cfg,
)


@dataclass
class StableContactMoveCfg:
  # Contact target (env frame: robot spawns at the env origin facing +x).
  active_hand: str = "left"
  wall_distance: float = 0.40
  """m from the env origin to the wall face, on the active hand's side."""
  wall_face_height: float = 1.0
  target_forward: float = 0.0
  """m along +x. 0 = beside the shoulder, which gives a symmetric reach window."""
  target_height: float = 0.95
  """m (world z) of the fixed palm target."""
  initial_preload: float = 0.01
  """m the initial palm pose is placed inside the face, so episodes start in contact."""

  # Contact reward / safety.
  target_std: float = 0.05
  """m, kernel width of the hand-target reward."""
  persistence_time: float = 0.5
  """s of unbroken contact for full persistence reward."""
  force_limit: float = 30.0
  """N, pressing force above which the excess-force penalty starts."""
  severe_force_limit: float = 80.0
  """N, termination."""
  max_target_error: float = 0.20
  """m, hand-target distance that ends the episode (contact abandoned)."""
  contact_loss_grace_s: float = 1.0
  """s without contact before termination; short breaks are tolerated."""

  # Motion command (parallel to the wall).
  motion_direction: tuple[float, float] = (1.0, 0.0)
  displacement_range: float = 0.15
  """m, final bound on the body target offset. Reach study: the pelvis can move
  about +-0.36 m with the palm fixed at the default target."""
  displacement_stages: tuple[tuple[int, float], ...] = (
    (0, 0.0), (500 * 24, 0.08), (1500 * 24, 0.15),
  )
  """(env step, range): stand -> small motion -> larger motion. 24 env steps per
  PPO iteration, so stages start at iterations 0, 500, 1500."""
  position_gain: float = 1.0
  max_speed: float = 0.2
  resampling_time_range: tuple[float, float] = (4.0, 8.0)
  rel_standing_envs: float = 0.2

  episode_length_s: float = 20.0

  force_band: tuple[float, float] | None = None
  """N, (f_min, f_max) of the hand_force_band reward (1 inside, Gaussian falloff
  outside). None = (0, force_limit). Only used if hand_force_band has a weight."""
  force_band_std: float = 1.0
  """N, falloff width of hand_force_band outside the band."""

  body_position_weight: float = 0.0
  """Weight of mdp.body_position_tracking (pelvis error along the command axis). 0 = off."""
  body_position_std: float = 0.06
  """m, kernel width of the body-position reward."""

  stance_position_weight: float = 0.0
  """Weight of mdp.stance_position_tracking (feet midpoint along the command axis). 0 = off."""
  stance_position_std: float = 0.06
  """m, kernel width of the stance-position reward."""
  stance_offset: float = 0.014
  """m, feet-midpoint minus pelvis offset along the command axis at reset (default stance)."""

  foot_position_weight: float = 0.0
  """Weight of mdp.foot_position_tracking (each foot's displacement along the command axis). 0 = off."""
  foot_position_std: float = 0.06
  """m, kernel width of the per-foot displacement reward."""
  foot_offsets: tuple[float, float] = (0.014, 0.014)
  """m, (left, right) foot-site position along the command axis minus the pelvis at reset."""

  gravity_hold_joints: tuple[str, ...] = ()
  """Joint regexes whose zero-action position target cancels gravity at the default
  pose (mdp.GravityHoldJointPositionAction). Empty = stock offset (V1 baseline)."""

  # Weights. Contact terms are new; motion/safety terms keep Unitree-G1-Flat values.
  contact_weights: dict[str, float] = field(default_factory=lambda: {
    "hand_target": 1.0,
    "hand_contact_persistence": 0.5,
    "hand_slip": -10.0,
    "hand_force_excess": -1.0,
    "unintended_contact": -1.0,
  })


# Inherited Unitree-G1-Flat reward terms that are dropped (see report / README).
REMOVED_VELOCITY_REWARDS = (
  "stand_still",  # Fights holding a displaced posture at zero command.
  "angular_momentum",  # Meant to shape free arm swing; the active arm is constrained.
  "body_ang_vel",  # Redundant with body_orientation_l2 for a slow task.
  "joint_acc_l2",  # Redundant with action_rate_l2; keep the set small.
  "soft_landing",  # Negligible weight (1e-3).
  "foot_clearance",  # Swing-height shaping; secondary at <= 0.2 m/s.
)


def wall_inward_normal(cfg: StableContactMoveCfg) -> tuple[float, float, float]:
  """Unit vector from the robot into the wall (env frame)."""
  return (0.0, 1.0 if cfg.active_hand == "left" else -1.0, 0.0)


def palm_target_w(cfg: StableContactMoveCfg) -> tuple[float, float, float]:
  """Initial palm-site position in the env frame (inside the face by the preload)."""
  side = wall_inward_normal(cfg)[1]
  lateral = cfg.wall_distance - (HAND_CAPSULE_RADIUS - cfg.initial_preload)
  return (cfg.target_forward, side * lateral, cfg.target_height)


def make_stable_contact_move_env_cfg(
  task: StableContactMoveCfg | None = None, play: bool = False
) -> ManagerBasedRlEnvCfg:
  task = task or StableContactMoveCfg()
  hand = G1_HANDS[task.active_hand]
  target_local = contact_target_local(
    task.active_hand, task.target_forward, task.target_height - task.wall_face_height
  )
  contact_rewards = make_contact_reward_terms(
    task.active_hand,
    task.contact_weights,
    force_range=task.force_band if task.force_band is not None else (0.0, task.force_limit),
    force_std=task.force_band_std,
    force_limit=task.force_limit,
    target_local=target_local,
    target_std=task.target_std,
    cone_margin=1.0,  # Unused: no friction-cone term in V1.
    persistence_time=task.persistence_time,
  )
  cfg = make_stable_contact_stand_env_cfg(
    active_hand=task.active_hand,
    wall_distance=task.wall_distance,
    wall_face_height=task.wall_face_height,
    target_forward=task.target_forward,
    target_height_offset=task.target_height - task.wall_face_height,
    contact_rewards=contact_rewards,
    play=play,
  )
  sensor = hand_sensor_name(task.active_hand)
  surface = {"surface_name": SURFACE, "normal_local": WALL_OUTWARD_NORMAL_LOCAL}

  def site_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", site_names=(hand.site,))

  # Initial state: feet on the floor (no reset drop) and the active arm holding
  # the palm on the target.
  height = standing_pelvis_height()
  arm = dict(contact_ready_arm_pose(
    task.active_hand, palm_target_w(task), height, wall_inward_normal(task)))
  robot = cfg.scene.entities["robot"]
  robot.init_state = EntityCfg.InitialStateCfg(
    pos=(HOME_KEYFRAME.pos[0], HOME_KEYFRAME.pos[1], height),
    rot=HOME_KEYFRAME.rot,
    joint_pos={**home_joint_positions(), **arm},
    joint_vel={".*": 0.0},
  )
  cfg.events.pop("push_robot", None)

  if task.gravity_hold_joints:
    base = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = mdp.GravityHoldJointPositionActionCfg(
      **{f.name: getattr(base, f.name) for f in fields(base) if f.init},
      hold_joint_names=task.gravity_hold_joints,
    )

  # Motion command: same name and (vx, vy, wz) layout as the velocity task.
  cfg.commands["twist"] = mdp.BaseDisplacementCommandCfg(
    entity_name="robot",
    direction=task.motion_direction,
    displacement_range=task.displacement_stages[-1][1] if play else task.displacement_stages[0][1],
    position_gain=task.position_gain,
    max_speed=task.max_speed,
    resampling_time_range=task.resampling_time_range,
    rel_standing_envs=task.rel_standing_envs,
  )
  assert task.displacement_stages[-1][1] == task.displacement_range
  cfg.curriculum = {} if play else {
    "displacement": CurriculumTermCfg(
      func=mdp.displacement_range_stages,
      params={"command_name": "twist", "stages": task.displacement_stages},
    )
  }

  # Rewards: audit of the inherited velocity-task terms.
  for name in REMOVED_VELOCITY_REWARDS:
    cfg.rewards.pop(name)
  pose = cfg.rewards["pose"].params
  pose["std_standing"] = dict(pose["std_walking"])  # Displaced holding postures.

  if task.body_position_weight:
    cfg.rewards["body_position"] = RewardTermCfg(
      func=mdp.body_position_tracking, weight=task.body_position_weight,
      params={"command_name": "twist", "std": task.body_position_std},
    )

  if task.stance_position_weight:
    cfg.rewards["stance_position"] = RewardTermCfg(
      func=mdp.stance_position_tracking, weight=task.stance_position_weight,
      params={"command_name": "twist", "std": task.stance_position_std, "offset": task.stance_offset,
              "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)},
    )

  if task.foot_position_weight:
    cfg.rewards["foot_position"] = RewardTermCfg(
      func=mdp.foot_position_tracking, weight=task.foot_position_weight,
      params={"command_name": "twist", "std": task.foot_position_std, "offsets": task.foot_offsets,
              "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)},
    )

  # Terminations (time_out and fell_over are inherited).
  cfg.terminations.update({
    "hand_contact_lost": TerminationTermCfg(
      func=mdp.hand_contact_lost,
      params={"sensor_name": sensor, "grace_period_s": task.contact_loss_grace_s},
    ),
    "hand_target_lost": TerminationTermCfg(
      func=mdp.hand_target_error_exceeded,
      params={"surface_name": SURFACE, "target_local": target_local,
              "max_error": task.max_target_error, "asset_cfg": site_cfg()},
    ),
    "hand_force_severe": TerminationTermCfg(
      func=mdp.hand_contact_force_exceeded,
      params={"sensor_name": sensor, **surface, "force_limit": task.severe_force_limit},
    ),
  })

  # Evaluation metrics.
  arm_joints = SceneEntityCfg("robot", joint_names=(hand.arm_joints_regex,))
  cfg.metrics.update({
    "command_speed": MetricsTermCfg(func=mdp.command_speed, params={"command_name": "twist"}),
    "base_speed": MetricsTermCfg(func=mdp.base_speed,
                                 params={"asset_cfg": SceneEntityCfg("robot")}),
    "hand_target_error": MetricsTermCfg(
      func=mdp.hand_target_error,
      params={"surface_name": SURFACE, "target_local": target_local, "asset_cfg": site_cfg()}),
    "hand_contact_ratio": MetricsTermCfg(
      func=mdp.hand_contact_ratio, params={"sensor_name": sensor}),
    "hand_normal_force": MetricsTermCfg(
      func=mdp.hand_normal_force, params={"sensor_name": sensor, **surface}),
    "hand_force_over_limit": MetricsTermCfg(
      func=mdp.hand_force_over_limit,
      params={"sensor_name": sensor, **surface, "force_limit": task.force_limit}),
    "hand_tangential_speed": MetricsTermCfg(
      func=mdp.hand_tangential_speed,
      params={"sensor_name": sensor, **surface, "asset_cfg": site_cfg()}),
    "hand_contact_breaks_per_s": MetricsTermCfg(
      func=mdp.hand_contact_breaks_per_s, params={"sensor_name": sensor}),
    "unintended_contact_count": MetricsTermCfg(
      func=mdp.unintended_contact_count, params={"sensor_name": UNINTENDED_SENSOR}),
    "active_arm_limit_margin": MetricsTermCfg(
      func=mdp.active_arm_limit_margin, params={"asset_cfg": arm_joints}),
  })

  cfg.episode_length_s = int(1e9) if play else task.episode_length_s
  return cfg

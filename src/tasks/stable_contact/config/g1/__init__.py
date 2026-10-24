from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from src.tasks.stable_contact.stable_contact_env_cfg import G1_HANDS
from src.tasks.stable_contact.stable_contact_move_env_cfg import (
  StableContactMoveCfg,
  make_stable_contact_move_env_cfg,
)

from .rl_cfg import (
  unitree_g1_stable_contact_actor_cmd_gain_ppo_runner_cfg,
  unitree_g1_stable_contact_ppo_runner_cfg,
)

# EXPERIMENTAL: fixed-point contact baseline (see stable_contact_move_env_cfg.py).
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove",
  env_cfg=make_stable_contact_move_env_cfg(),
  play_env_cfg=make_stable_contact_move_env_cfg(play=True),
  rl_cfg=unitree_g1_stable_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# EXPERIMENTAL A/B variant: identical to the baseline except that zero action
# holds the active arm and the waist against gravity at the reset pose
# (mdp.GravityHoldJointPositionAction).
_GRAV_HOLD = StableContactMoveCfg(
  gravity_hold_joints=(G1_HANDS[StableContactMoveCfg().active_hand].arm_joints_regex, r"waist_.*")
)
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove-GravHold",
  env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD),
  play_env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD, play=True),
  rl_cfg=unitree_g1_stable_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# EXPERIMENTAL A/B variant of GravHold: identical except hand_slip weight -10 -> -1
# (the contact-only slip penalty made touching the wall reward-suboptimal under
# exploration noise).
_GRAV_HOLD_SLIP1 = StableContactMoveCfg(
  gravity_hold_joints=_GRAV_HOLD.gravity_hold_joints,
  contact_weights={**StableContactMoveCfg().contact_weights, "hand_slip": -1.0},
)
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove-GravHold-Slip1",
  env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_SLIP1),
  play_env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_SLIP1, play=True),
  rl_cfg=unitree_g1_stable_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# EXPERIMENTAL A/B variant of GravHold-Slip1: identical except the existing
# binary hand_contact reward is enabled at +4 (Slip1's measured contact-vs-hover
# deficit was ~-0.054/step; +4 * dt = +0.08/step reverses it).
_GRAV_HOLD_CONTACT4 = StableContactMoveCfg(
  gravity_hold_joints=_GRAV_HOLD.gravity_hold_joints,
  contact_weights={**_GRAV_HOLD_SLIP1.contact_weights, "hand_contact": 4.0},
)
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove-GravHold-Contact4",
  env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_CONTACT4),
  play_env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_CONTACT4, play=True),
  rl_cfg=unitree_g1_stable_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# EXPERIMENTAL A/B variant of GravHold-Contact4: the binary hand_contact (+4) is
# replaced by the bounded hand_force_band (+4): 1 for 5-15 N pressing, Gaussian
# falloff (std 5 N) outside, 0 without contact. Force thresholds are unchanged.
_GRAV_HOLD_FORCE_BAND = StableContactMoveCfg(
  gravity_hold_joints=_GRAV_HOLD.gravity_hold_joints,
  contact_weights={**_GRAV_HOLD_SLIP1.contact_weights, "hand_force_band": 4.0},
  force_band=(5.0, 15.0),
  force_band_std=5.0,
)
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove-GravHold-ForceBand",
  env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND),
  play_env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND, play=True),
  rl_cfg=unitree_g1_stable_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# EXPERIMENTAL movement stage 1 of GravHold-ForceBand: identical except the body
# displacement command starts directly at +-0.08 m (no stationary stage, no 0.15 m
# stage). Used to warm-start from a validated stationary ForceBand checkpoint, since
# the step-based curriculum restarts from env step 0 in a resumed run.
_GRAV_HOLD_FORCE_BAND_MOVE08 = StableContactMoveCfg(
  gravity_hold_joints=_GRAV_HOLD_FORCE_BAND.gravity_hold_joints,
  contact_weights=_GRAV_HOLD_FORCE_BAND.contact_weights,
  force_band=_GRAV_HOLD_FORCE_BAND.force_band,
  force_band_std=_GRAV_HOLD_FORCE_BAND.force_band_std,
  displacement_range=0.08,
  displacement_stages=((0, 0.08),),
)
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove-GravHold-ForceBand-Move08",
  env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08),
  play_env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08, play=True),
  rl_cfg=unitree_g1_stable_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# EXPERIMENTAL A/B variant of Move08: identical except the displacement command's
# position_gain (1.0 -> 2.5 /s). With gain 1.0 a <= 0.08 m position error never
# commands more than 0.08 m/s, below the gait reward's 0.1 m/s activation threshold.
_GRAV_HOLD_FORCE_BAND_MOVE08_GAIN2P5 = StableContactMoveCfg(
  gravity_hold_joints=_GRAV_HOLD_FORCE_BAND_MOVE08.gravity_hold_joints,
  contact_weights=_GRAV_HOLD_FORCE_BAND_MOVE08.contact_weights,
  force_band=_GRAV_HOLD_FORCE_BAND_MOVE08.force_band,
  force_band_std=_GRAV_HOLD_FORCE_BAND_MOVE08.force_band_std,
  displacement_range=_GRAV_HOLD_FORCE_BAND_MOVE08.displacement_range,
  displacement_stages=_GRAV_HOLD_FORCE_BAND_MOVE08.displacement_stages,
  position_gain=2.5,
)
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-Gain2p5",
  env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_GAIN2P5),
  play_env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_GAIN2P5, play=True),
  rl_cfg=unitree_g1_stable_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# EXPERIMENTAL A/B variant of Move08-Gain2p5: identical plus a direct pelvis-position
# tracking reward along the command axis (weight 1.0, std 0.06 m) toward the
# displacement command's own world-frame target.
_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK = StableContactMoveCfg(
  gravity_hold_joints=_GRAV_HOLD_FORCE_BAND_MOVE08_GAIN2P5.gravity_hold_joints,
  contact_weights=_GRAV_HOLD_FORCE_BAND_MOVE08_GAIN2P5.contact_weights,
  force_band=_GRAV_HOLD_FORCE_BAND_MOVE08_GAIN2P5.force_band,
  force_band_std=_GRAV_HOLD_FORCE_BAND_MOVE08_GAIN2P5.force_band_std,
  displacement_range=_GRAV_HOLD_FORCE_BAND_MOVE08_GAIN2P5.displacement_range,
  displacement_stages=_GRAV_HOLD_FORCE_BAND_MOVE08_GAIN2P5.displacement_stages,
  position_gain=_GRAV_HOLD_FORCE_BAND_MOVE08_GAIN2P5.position_gain,
  body_position_weight=1.0,
  body_position_std=0.06,
)
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack",
  env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK),
  play_env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK, play=True),
  rl_cfg=unitree_g1_stable_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# EXPERIMENTAL A/B variant of Move08-PosTrack: identical plus a feet-midpoint
# (stance) displacement reward toward the same commanded displacement
# (weight 1.0, std 0.06 m, reset offset 0.014 m).
_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE = StableContactMoveCfg(
  gravity_hold_joints=_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK.gravity_hold_joints,
  contact_weights=_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK.contact_weights,
  force_band=_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK.force_band,
  force_band_std=_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK.force_band_std,
  displacement_range=_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK.displacement_range,
  displacement_stages=_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK.displacement_stages,
  position_gain=_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK.position_gain,
  body_position_weight=_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK.body_position_weight,
  body_position_std=_GRAV_HOLD_FORCE_BAND_MOVE08_POSTRACK.body_position_std,
  stance_position_weight=1.0,
  stance_position_std=0.06,
  stance_offset=0.014,
)
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack",
  env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE),
  play_env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE, play=True),
  rl_cfg=unitree_g1_stable_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# EXPERIMENTAL A/B variant of Move08-PosTrack-StanceTrack: identical plus a per-foot
# commanded-displacement reward (weight 1.0, std 0.06 m; each foot keeps its own
# reset reference, both at +0.014 m along the command axis).
_GRAV_HOLD_FORCE_BAND_MOVE08_FOOT = StableContactMoveCfg(
  gravity_hold_joints=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.gravity_hold_joints,
  contact_weights=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.contact_weights,
  force_band=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.force_band,
  force_band_std=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.force_band_std,
  displacement_range=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.displacement_range,
  displacement_stages=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.displacement_stages,
  position_gain=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.position_gain,
  body_position_weight=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.body_position_weight,
  body_position_std=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.body_position_std,
  stance_position_weight=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.stance_position_weight,
  stance_position_std=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.stance_position_std,
  stance_offset=_GRAV_HOLD_FORCE_BAND_MOVE08_STANCE.stance_offset,
  foot_position_weight=1.0,
  foot_position_std=0.06,
  foot_offsets=(0.014, 0.014),
)
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack-FootTrack",
  env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_FOOT),
  play_env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_FOOT, play=True),
  rl_cfg=unitree_g1_stable_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# EXPERIMENTAL A/B variant of FootTrack: identical environment (real command, rewards, gates,
# critic); only the actor sees its translational (vx, vy) command input as clip_norm(1.5 * v, 0.2).
# The actor normalizer keeps updating on the real command, so the gain is not absorbed.
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack-FootTrack-ActorCmd1p5",
  env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_FOOT),
  play_env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_FOOT, play=True),
  rl_cfg=unitree_g1_stable_contact_actor_cmd_gain_ppo_runner_cfg(
    "src.tasks.stable_contact.actor_command_gain:ActorCommandGain1p5"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# EXPERIMENTAL A/B variant of FootTrack-ActorCmd1p5: identical except that the ACTOR's translational
# command uses a pelvis/stance blended error along the command axis (alpha 0.25; perpendicular error
# stays the pelvis error). Command manager, rewards, gates and critic keep the real command.
_GRAV_HOLD_FORCE_BAND_MOVE08_FOOT_STANCEREF = replace(_GRAV_HOLD_FORCE_BAND_MOVE08_FOOT, actor_command_stance_alpha=0.25)
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack-FootTrack-ActorCmd1p5-StanceRefP025",
  env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_FOOT_STANCEREF),
  play_env_cfg=make_stable_contact_move_env_cfg(_GRAV_HOLD_FORCE_BAND_MOVE08_FOOT_STANCEREF, play=True),
  rl_cfg=unitree_g1_stable_contact_actor_cmd_gain_ppo_runner_cfg(
    "src.tasks.stable_contact.actor_command_gain:ActorCommandGain1p5"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

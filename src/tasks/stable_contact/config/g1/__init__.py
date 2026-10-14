from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from src.tasks.stable_contact.stable_contact_env_cfg import G1_HANDS
from src.tasks.stable_contact.stable_contact_move_env_cfg import (
  StableContactMoveCfg,
  make_stable_contact_move_env_cfg,
)

from .rl_cfg import unitree_g1_stable_contact_ppo_runner_cfg

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

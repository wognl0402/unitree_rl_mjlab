"""CPU tests for the EXPERIMENTAL Unitree-G1-StableContactMove task.

Run: python -m pytest tests/test_stable_contact_move.py -q
"""

import copy
import math
import re
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
import src.tasks.stable_contact.mdp as mdp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from src.tasks.stable_contact.contact_pose import (
  _model,
  _set_spawn,
  contact_ready_arm_pose,
  hand_capsule_depth,
  home_joint_positions,
  standing_pelvis_height,
)
from src.tasks.stable_contact.stable_contact_env_cfg import SURFACE, hand_sensor_name
from src.tasks.stable_contact.stable_contact_move_env_cfg import (
  REMOVED_VELOCITY_REWARDS,
  StableContactMoveCfg,
  make_stable_contact_move_env_cfg,
  palm_target_w,
  wall_inward_normal,
)

TASK = "Unitree-G1-StableContactMove"
NUM_JOINTS = 29


def _build(task=None, num_envs=1):
  cfg = make_stable_contact_move_env_cfg(task)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  return env


@pytest.fixture(scope="module")
def env():
  e = _build(num_envs=2)
  yield e
  e.close()


def _target_w(env):
  params = env.reward_manager.get_term_cfg("hand_target").params
  return mdp.surface_point_w(env, SURFACE, params["target_local"])


def _palm_w(env, hand="left"):
  robot = env.scene["robot"]
  return robot.data.site_pos_w[:, robot.find_sites(f"{hand}_palm")[0][0]]


##
# Registration and isolation.
##


def test_task_registered_with_separate_experiment():
  assert TASK in list_tasks()
  assert load_rl_cfg(TASK).experiment_name == "g1_stable_contact_move"
  assert load_rl_cfg("Unitree-G1-Flat").experiment_name == "g1_velocity"
  cfg = load_env_cfg(TASK)
  assert isinstance(cfg.commands["twist"], mdp.BaseDisplacementCommandCfg)


def test_baseline_flat_task_unchanged():
  flat = load_env_cfg("Unitree-G1-Flat")
  assert "wall" not in flat.scene.entities
  assert flat.scene.entities["robot"].init_state.pos == (0, 0, 0.8)
  for name in REMOVED_VELOCITY_REWARDS:
    assert name in flat.rewards
  assert "hand_target" not in flat.rewards
  assert set(flat.terminations) == {"time_out", "fell_over"}


##
# Reward / termination composition.
##


def test_reward_set_is_minimal_and_audited(env):
  active = set(env.reward_manager.active_terms)
  assert active == {
    "track_linear_velocity", "track_angular_velocity", "body_orientation_l2", "pose",
    "is_terminated", "joint_pos_limits", "action_rate_l2", "foot_gait", "foot_slip",
    "self_collisions", "hand_target", "hand_contact_persistence", "hand_slip",
    "hand_force_excess", "unintended_contact",
  }
  assert not active & set(REMOVED_VELOCITY_REWARDS)
  # Fixed point: slip is measured against zero desired hand velocity.
  assert env.reward_manager.get_term_cfg("hand_slip").params.get("command_name") is None


def test_terminations(env):
  assert set(env.termination_manager.active_terms) == {
    "time_out", "fell_over", "hand_contact_lost", "hand_target_lost", "hand_force_severe",
  }


def test_active_arm_free_inactive_arm_regularized(env):
  robot = env.scene["robot"]
  pose = env.reward_manager.get_term_cfg("pose").params
  names = {robot.joint_names[i] for i in pose["asset_cfg"].joint_ids}
  assert not any(n.startswith(("left_shoulder", "left_elbow", "left_wrist")) for n in names)
  assert {"right_shoulder_pitch_joint", "right_elbow_joint", "left_knee_joint"} <= names
  assert pose["std_standing"] == pose["std_walking"]


##
# Contact-ready initial state.
##


@pytest.mark.parametrize("hand", ["left", "right"])
def test_contact_ready_pose_puts_flat_palm_on_target(hand):
  task = StableContactMoveCfg(active_hand=hand)
  height = standing_pelvis_height()
  arm = dict(contact_ready_arm_pose(hand, palm_target_w(task), height, wall_inward_normal(task)))
  m = _model()
  d = mujoco.MjData(m)
  _set_spawn(d, m, {**home_joint_positions(), **arm}, height)
  mujoco.mj_kinematics(m, d)
  assert np.allclose(d.site(f"{hand}_palm").xpos, palm_target_w(task), atol=1e-3)
  normal = np.array(wall_inward_normal(task))
  depth = hand_capsule_depth(m, d, hand, normal * task.wall_distance, normal)
  assert 0.0 <= depth <= task.initial_preload + 0.003  # Light, uniform preload.
  for name, q in arm.items():
    lo, hi = m.jnt_range[m.joint(name).id]
    c, r = (lo + hi) / 2, (hi - lo) / 2
    assert c - 0.9 * r <= q <= c + 0.9 * r


def test_episode_starts_in_light_contact_at_standing_height(env):
  sensor = hand_sensor_name("left")
  assert torch.norm(_palm_w(env) - _target_w(env), dim=-1).max() < 0.02
  obs, *_ = env.step(torch.zeros(2, NUM_JOINTS))
  assert torch.isfinite(obs["actor"]).all()
  assert bool(mdp.hand_in_contact(env, sensor).all())
  f = mdp.hand_contact_force(env, sensor, SURFACE, (-1.0, 0.0, 0.0))[:, 0]
  assert (f > 0).all() and (f < StableContactMoveCfg().force_limit).all()
  base_z = env.scene["robot"].data.root_link_pos_w[:, 2]
  assert torch.allclose(base_z, torch.full_like(base_z, standing_pelvis_height()), atol=0.01)


##
# Fixed world-space target.
##


def test_target_fixed_while_body_moves():
  env = _build()
  target0 = _target_w(env).clone()
  error0 = torch.norm(_palm_w(env) - target0, dim=-1)
  rng = torch.Generator().manual_seed(0)
  for _ in range(30):
    env.step(torch.rand(1, NUM_JOINTS, generator=rng) * 2 - 1)
    assert torch.equal(_target_w(env), target0)
  # Translate the whole robot 10 cm along the wall: the hand moves, the target does not.
  robot = env.scene["robot"]
  state = torch.cat([robot.data.root_link_pose_w, torch.zeros(1, 6)], dim=-1)
  env.reset()
  palm_before = _palm_w(env).clone()
  state = torch.cat([robot.data.root_link_pose_w, torch.zeros(1, 6)], dim=-1)
  state[:, 0] += 0.10
  robot.write_root_state_to_sim(state)
  env.sim.forward()
  assert torch.equal(_target_w(env), target0)
  assert torch.allclose(_palm_w(env) - palm_before, torch.tensor([[0.10, 0.0, 0.0]]), atol=1e-4)
  error1 = torch.norm(_palm_w(env) - target0, dim=-1)
  assert error1.item() == pytest.approx(math.hypot(0.10, error0.item()), abs=0.01)
  env.close()


def test_reset_restores_target_pose_and_contact_state():
  env = _build()
  sensor = env.scene[hand_sensor_name("left")]
  target0 = _target_w(env).clone()
  for _ in range(40):
    env.step(torch.ones(1, NUM_JOINTS))  # Drive the robot away from the wall.
  env.reset()
  assert torch.equal(_target_w(env), target0)
  assert torch.norm(_palm_w(env) - target0, dim=-1).item() < 0.02
  assert float(sensor.data.current_contact_time[0, 0]) == 0.0
  assert float(sensor.data.current_air_time[0, 0]) == 0.0
  base = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
  assert torch.allclose(base[:, :2], torch.zeros(1, 2), atol=1e-4)
  env.close()


##
# Wrong hand / wrong wall.
##


def test_wrong_hand_does_not_satisfy_target(env):
  params = dict(env.reward_manager.get_term_cfg("hand_target").params)
  left = mdp.hand_target_tracking(env, **params)
  params["asset_cfg"] = SceneEntityCfg("robot", site_names=("right_palm",))
  params["asset_cfg"].resolve(env.scene)
  right = mdp.hand_target_tracking(env, **params)
  assert (left > 0.8).all() and (right < 1e-6).all()
  assert not mdp.hand_in_contact(env, hand_sensor_name("right")).any()


def test_contact_counts_only_the_configured_wall(env):
  sensor = env.scene[hand_sensor_name("left")]
  assert sensor.cfg.secondary.entity == SURFACE
  assert sensor.cfg.secondary.pattern == "wall_geom"
  assert sensor.cfg.primary.pattern == "left_hand_collision"
  unintended = env.scene["robot_wall_unintended_contact"].cfg
  assert "left_hand_collision" in unintended.primary.exclude
  assert unintended.secondary.entity == SURFACE


##
# Slip and force on stub data, using the task's own reward parameters.
##


def _stub(force_w, palm_vel_w, found=1.0):
  yaw = math.pi / 2
  wall = SimpleNamespace(data=SimpleNamespace(
    root_link_quat_w=torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]),
    root_link_pos_w=torch.tensor([[0.0, 0.4, 1.0]]),
  ))
  sensor = SimpleNamespace(cfg=SimpleNamespace(reduce="netforce"), data=SimpleNamespace(
    found=torch.tensor([[found]]), force=torch.tensor([[force_w]])))
  robot = SimpleNamespace(data=SimpleNamespace(site_lin_vel_w=torch.tensor([[palm_vel_w]])))
  return SimpleNamespace(num_envs=1, scene={SURFACE: wall, "s": sensor, "robot": robot})


def _task_params(name):
  params = dict(make_stable_contact_move_env_cfg().rewards[name].params)
  params["sensor_name"] = "s"
  if "asset_cfg" in params:
    params["asset_cfg"] = SimpleNamespace(name="robot", site_ids=[0])
  return params


def test_fixed_point_slip_penalty():
  params = _task_params("hand_slip")
  still = mdp.hand_slip(_stub([0, 5, 0], [0, 0, 0]), **params)
  pressing_in = mdp.hand_slip(_stub([0, 5, 0], [0, 0.2, 0]), **params)  # Along the normal.
  sliding = mdp.hand_slip(_stub([0, 5, 0], [0.2, 0, 0]), **params)  # Along the wall.
  faster = mdp.hand_slip(_stub([0, 5, 0], [0.4, 0, 0]), **params)
  assert still.item() == 0.0 and pressing_in.item() == pytest.approx(0.0, abs=1e-9)
  assert 0.0 < sliding.item() < faster.item()


def test_excess_force_penalty_activates_above_limit():
  params = _task_params("hand_force_excess")
  limit = StableContactMoveCfg().force_limit
  below = mdp.hand_contact_force_excess(_stub([0, limit - 1, 0], [0, 0, 0]), **params)
  above = mdp.hand_contact_force_excess(_stub([0, 2 * limit, 0], [0, 0, 0]), **params)
  assert below.item() == 0.0
  assert above.item() == pytest.approx(1.0)


##
# Motion command.
##


def test_displacement_command_is_bounded_and_points_to_target():
  task = StableContactMoveCfg(rel_standing_envs=0.0)
  cfg = make_stable_contact_move_env_cfg(task)
  cfg.scene.num_envs = 64
  cfg.curriculum = {}
  cfg.commands["twist"].displacement_range = task.displacement_range
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  term = env.command_manager.get_term("twist")
  offset = term.target_pos_w - env.scene.env_origins[:, :2]
  assert (offset[:, 1].abs() < 1e-6).all()  # Parallel to the wall only.
  assert (offset[:, 0].abs() <= task.displacement_range + 1e-6).all()
  assert offset[:, 0].abs().max() > 0.5 * task.displacement_range
  env.step(torch.zeros(64, NUM_JOINTS))
  cmd = env.command_manager.get_command("twist")
  speed = torch.norm(cmd[:, :2], dim=-1)
  assert (speed <= task.max_speed + 1e-6).all()
  error = term.target_pos_w - env.scene["robot"].data.root_link_pos_w[:, :2]
  far = torch.norm(error, dim=-1) > 0.05
  assert (torch.sign(cmd[far, 0]) == torch.sign(error[far, 0])).all()
  env.close()


def test_command_is_zero_at_target_and_curriculum_stages():
  env = _build()
  term = env.command_manager.get_term("twist")
  term.target_pos_w[:] = env.scene["robot"].data.root_link_pos_w[:, :2]
  term._update_command()
  assert torch.norm(term.command[:, :2]).item() < 1e-6
  stages = StableContactMoveCfg().displacement_stages
  for step, expected in stages:
    env.common_step_counter = step
    mdp.displacement_range_stages(env, None, "twist", stages)
    assert term.cfg.displacement_range == expected
  env.close()


@pytest.mark.parametrize("hand", ["left", "right"])
def test_command_range_is_reachable_with_margin(hand):
  """With the palm fixed, the pelvis can sit anywhere within twice the final
  displacement range and the flat palm is still reachable inside soft limits."""
  task = StableContactMoveCfg(active_hand=hand)
  height = standing_pelvis_height()
  tx, ty, tz = palm_target_w(task)
  for d in np.linspace(-2 * task.displacement_range, 2 * task.displacement_range, 9):
    # Moving the pelvis by +d equals moving the fixed target by -d in its frame.
    contact_ready_arm_pose(hand, (tx - float(d), ty, tz), height, wall_inward_normal(task))


##
# Rollout sanity.
##


def test_random_rollout_rewards_and_metrics_finite():
  """Evaluate raw reward functions: the reward manager replaces NaN/Inf with 0,
  so checking its output alone could not detect a broken term."""
  env = _build()
  rng = torch.Generator().manual_seed(1)
  rm = env.reward_manager
  for _ in range(100):
    _, reward, *_ = env.step(torch.rand(1, NUM_JOINTS, generator=rng) * 2 - 1)
    assert torch.isfinite(reward).all()
    for name in rm.active_terms:
      term = rm.get_term_cfg(name)
      assert torch.isfinite(term.func(env, **term.params)).all(), name
    assert torch.isfinite(env.metrics_manager._step_values).all()
  env.close()


def test_default_action_offset_is_stock_baseline(env):
  """Gravity hold is opt-in: the registered task keeps the V1 joint-position offset."""
  term = env.action_manager.get_term("joint_pos")
  assert type(term) is mdp.actions.JointPositionAction
  robot = env.scene["robot"]
  assert torch.allclose(term.offset, robot.data.default_joint_pos[:, term.target_ids])


def test_gravity_hold_offset_cancels_gravity_at_default_pose():
  """offset - default = qfrc_bias / kp on the held joints (position servos, qvel = 0)."""
  arm = r"left_(shoulder|elbow|wrist)_.*"
  env = _build(StableContactMoveCfg(gravity_hold_joints=(arm, r"waist_.*")))
  term = env.action_manager.get_term("joint_pos")
  robot = env.scene["robot"]
  m = env.sim.mj_model
  d = mujoco.MjData(m)
  root = robot.data.default_root_state[0].cpu().numpy()
  free = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
              and m.joint(j).name.startswith("robot/"))
  a = m.jnt_qposadr[free]
  d.qpos[a:a + 3] = (0.0, 0.0, root[2])
  d.qpos[a + 3:a + 7] = root[3:7]
  for name, q in zip(robot.joint_names, robot.data.default_joint_pos[0].tolist()):
    d.qpos[m.jnt_qposadr[m.joint(f"robot/{name}").id]] = q
  mujoco.mj_forward(m, d)
  delta = (term.offset - robot.data.default_joint_pos[:, term.target_ids])[0]
  held = 0
  for i, name in enumerate(term.target_names):
    full = f"robot/{name}"
    act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
    expect = d.qfrc_bias[m.jnt_dofadr[m.joint(full).id]] / m.actuator_gainprm[act, 0]
    if re.fullmatch(arm, name) or name.startswith("waist_"):
      held += 1
      assert float(delta[i]) == pytest.approx(expect, abs=1e-5)
    else:
      assert float(delta[i]) == 0.0
  assert held == 10
  # The shoulder carries the abducted arm: the correction is not negligible.
  assert float(delta[term.target_names.index("left_shoulder_pitch_joint")]) > 0.1
  env.close()


def test_gravhold_task_differs_from_baseline_only_in_action_offset():
  """A/B control: the GravHold variant changes nothing but the joint_pos action term."""
  from dataclasses import fields as dc_fields

  def same(x, y):  # repr without per-call function addresses
    return re.sub(r" at 0x[0-9a-f]+", "", repr(x)) == re.sub(r" at 0x[0-9a-f]+", "", repr(y))

  base, grav = load_env_cfg(TASK), load_env_cfg("Unitree-G1-StableContactMove-GravHold")
  for f in dc_fields(base):
    if f.name != "actions":
      assert same(getattr(base, f.name), getattr(grav, f.name)), f.name
  a, b = base.actions["joint_pos"], grav.actions["joint_pos"]
  assert type(b) is mdp.GravityHoldJointPositionActionCfg
  for f in dc_fields(a):
    assert getattr(a, f.name) == getattr(b, f.name), f.name
  assert b.hold_joint_names == (r"left_(shoulder|elbow|wrist)_.*", r"waist_.*")
  assert same(load_rl_cfg(TASK), load_rl_cfg("Unitree-G1-StableContactMove-GravHold"))


def test_slip1_task_differs_from_gravhold_only_in_slip_weight():
  """A/B control: GravHold-Slip1 == GravHold except rewards.hand_slip.weight (-10 -> -1)."""
  from dataclasses import fields as dc_fields

  def same(x, y):  # repr without per-call function addresses
    return re.sub(r" at 0x[0-9a-f]+", "", repr(x)) == re.sub(r" at 0x[0-9a-f]+", "", repr(y))

  grav = load_env_cfg("Unitree-G1-StableContactMove-GravHold")
  slip1 = load_env_cfg("Unitree-G1-StableContactMove-GravHold-Slip1")
  for f in dc_fields(grav):
    if f.name != "rewards":
      assert same(getattr(grav, f.name), getattr(slip1, f.name)), f.name
  assert grav.rewards.keys() == slip1.rewards.keys()
  for name in grav.rewards:
    a, b = grav.rewards[name], slip1.rewards[name]
    if name == "hand_slip":
      assert (a.weight, b.weight) == (-10.0, -1.0)
      assert a.func is b.func and same(a.params, b.params)
    else:
      assert same(a, b), name
  assert same(load_rl_cfg("Unitree-G1-StableContactMove-GravHold"),
              load_rl_cfg("Unitree-G1-StableContactMove-GravHold-Slip1"))


def test_contact4_task_differs_from_slip1_only_by_hand_contact_term():
  """A/B control: GravHold-Contact4 == GravHold-Slip1 plus rewards.hand_contact (weight +4)."""
  from dataclasses import fields as dc_fields

  def same(x, y):  # repr without per-call function addresses
    return re.sub(r" at 0x[0-9a-f]+", "", repr(x)) == re.sub(r" at 0x[0-9a-f]+", "", repr(y))

  slip1 = load_env_cfg("Unitree-G1-StableContactMove-GravHold-Slip1")
  c4 = load_env_cfg("Unitree-G1-StableContactMove-GravHold-Contact4")
  for f in dc_fields(slip1):
    if f.name != "rewards":
      assert same(getattr(slip1, f.name), getattr(c4, f.name)), f.name
  assert set(c4.rewards) - set(slip1.rewards) == {"hand_contact"}
  assert set(slip1.rewards) - set(c4.rewards) == set()
  for name in slip1.rewards:
    assert same(slip1.rewards[name], c4.rewards[name]), name
  assert slip1.rewards["hand_slip"].weight == -1.0
  term = c4.rewards["hand_contact"]
  assert term.weight == 4.0 and term.func is mdp.hand_contact
  assert term.params == {"sensor_name": hand_sensor_name("left")}
  assert same(load_rl_cfg("Unitree-G1-StableContactMove-GravHold-Slip1"),
              load_rl_cfg("Unitree-G1-StableContactMove-GravHold-Contact4"))


def test_forceband_task_differs_from_contact4_only_in_contact_reward():
  """A/B control: ForceBand == Contact4 with hand_contact (+4) replaced by hand_force_band (+4)."""
  from dataclasses import fields as dc_fields

  def same(x, y):  # repr without per-call function addresses
    return re.sub(r" at 0x[0-9a-f]+", "", repr(x)) == re.sub(r" at 0x[0-9a-f]+", "", repr(y))

  c4 = load_env_cfg("Unitree-G1-StableContactMove-GravHold-Contact4")
  fb = load_env_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand")
  for f in dc_fields(c4):
    if f.name != "rewards":
      assert same(getattr(c4, f.name), getattr(fb, f.name)), f.name
  assert set(c4.rewards) - set(fb.rewards) == {"hand_contact"}
  assert set(fb.rewards) - set(c4.rewards) == {"hand_force_band"}
  for name in set(c4.rewards) & set(fb.rewards):
    assert same(c4.rewards[name], fb.rewards[name]), name
  term = fb.rewards["hand_force_band"]
  assert term.weight == 4.0 and term.func is mdp.hand_contact_force_band
  assert term.params["force_range"] == (5.0, 15.0) and term.params["std"] == 5.0
  assert term.params["sensor_name"] == hand_sensor_name("left")
  assert c4.rewards["hand_contact"].weight == 4.0
  # Thresholds of the force penalty / termination are untouched.
  assert fb.rewards["hand_force_excess"].params["force_limit"] == 30.0
  assert fb.terminations["hand_force_severe"].params["force_limit"] == 80.0
  assert same(load_rl_cfg("Unitree-G1-StableContactMove-GravHold-Contact4"),
              load_rl_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand"))


def test_force_band_reward_shape():
  """1 inside the band, exp(-(gap/std)^2) outside, 0 without contact."""
  env = SimpleNamespace()
  import src.tasks.stable_contact.mdp.rewards as rw
  forces = torch.tensor([0.0, 2.0, 10.0, 20.0, 25.0, 40.0])
  contact = torch.tensor([False, True, True, True, True, True])
  orig = (rw.surface_normal_w, rw.hand_contact_force_w, rw.hand_in_contact)
  try:
    rw.surface_normal_w = lambda *a: torch.tensor([[0.0, -1.0, 0.0]]).expand(6, 3)
    rw.hand_contact_force_w = lambda *a: torch.stack([torch.zeros(6), forces, torch.zeros(6)], -1)
    rw.hand_in_contact = lambda *a: contact
    r = rw.hand_contact_force_band(env, "s", "wall", (-1.0, 0.0, 0.0), (5.0, 15.0), 5.0)
  finally:
    rw.surface_normal_w, rw.hand_contact_force_w, rw.hand_in_contact = orig
  expect = [0.0, math.exp(-(3 / 5) ** 2), 1.0, math.exp(-1.0), math.exp(-4.0), math.exp(-25.0)]
  assert torch.allclose(r, torch.tensor(expect), atol=1e-6)


def test_move08_task_differs_from_forceband_only_in_displacement_command():
  """Movement stage 1: ForceBand with the +-0.08 m displacement range active from step 0."""
  from dataclasses import fields as dc_fields

  def same(x, y):  # repr without per-call function addresses
    return re.sub(r" at 0x[0-9a-f]+", "", repr(x)) == re.sub(r" at 0x[0-9a-f]+", "", repr(y))

  fb = load_env_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand")
  mv = load_env_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08")
  for f in dc_fields(fb):
    if f.name not in ("commands", "curriculum"):
      assert same(getattr(fb, f.name), getattr(mv, f.name)), f.name
  a, b = fb.commands["twist"], mv.commands["twist"]
  for f in dc_fields(a):
    if f.name != "displacement_range":
      assert getattr(a, f.name) == getattr(b, f.name), f.name
  assert (a.displacement_range, b.displacement_range) == (0.0, 0.08)
  assert fb.curriculum["displacement"].params["stages"] == ((0, 0.0), (500 * 24, 0.08), (1500 * 24, 0.15))
  assert mv.curriculum["displacement"].params["stages"] == ((0, 0.08),)
  assert same(load_rl_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand"),
              load_rl_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08"))


def test_move08_gain2p5_differs_from_move08_only_in_position_gain():
  """A/B control: Move08-Gain2p5 == Move08 except commands.twist.position_gain (1.0 -> 2.5)."""
  from dataclasses import fields as dc_fields

  def same(x, y):  # repr without per-call function addresses
    return re.sub(r" at 0x[0-9a-f]+", "", repr(x)) == re.sub(r" at 0x[0-9a-f]+", "", repr(y))

  g1 = load_env_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08")
  g2 = load_env_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-Gain2p5")
  for f in dc_fields(g1):
    if f.name != "commands":
      assert same(getattr(g1, f.name), getattr(g2, f.name)), f.name
  assert g1.commands.keys() == g2.commands.keys()
  a, b = g1.commands["twist"], g2.commands["twist"]
  for f in dc_fields(a):
    if f.name != "position_gain":
      assert getattr(a, f.name) == getattr(b, f.name), f.name
  assert (a.position_gain, b.position_gain) == (1.0, 2.5)
  assert b.max_speed == 0.2 and b.displacement_range == 0.08
  assert same(load_rl_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08"),
              load_rl_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-Gain2p5"))


# --- Body-position tracking reward (Move08-PosTrack) ---

def _fake_pos_env(target_xy, base_xy, direction=(1.0, 0.0)):
  d = torch.tensor(direction)
  term = SimpleNamespace(target_pos_w=torch.tensor(target_xy, dtype=torch.float32), direction=d / d.norm())
  base = torch.tensor(base_xy, dtype=torch.float32)
  robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.cat([base, torch.full((len(base), 1), 0.78)], dim=1)))
  return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: term), scene={"robot": robot})


def test_body_position_reward_max_at_zero_monotone_symmetric():
  errs = torch.tensor([0.0, 0.01, 0.02, 0.04, 0.06, 0.08])
  for sign in (1.0, -1.0):
    env = _fake_pos_env([[sign * e.item(), 0.0] for e in errs], [[0.0, 0.0]] * len(errs))
    r = mdp.body_position_tracking(env, "twist", std=0.06)
    assert r.shape == (len(errs),) and torch.all(torch.isfinite(r))
    assert r[0] == pytest.approx(1.0)
    assert torch.all(r[1:] < r[:-1])  # strictly decreasing with |error|
    if sign > 0:
      pos = r
  assert torch.allclose(pos, r)  # + and - errors give the same reward
  assert pos[-1] == pytest.approx(math.exp(-(0.08 / 0.06) ** 2), abs=1e-6)


def test_body_position_reward_ignores_off_axis_and_handles_zero_command():
  # Off-axis (wall-normal) and vertical pelvis offsets are not rewarded or penalized.
  env = _fake_pos_env([[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.05], [0.0, -0.05]])
  assert torch.allclose(mdp.body_position_tracking(env, "twist", std=0.06), torch.ones(2))
  # Zero command: target at the origin; standing there is maximal.
  env = _fake_pos_env([[0.0, 0.0]], [[0.0, 0.0]])
  assert mdp.body_position_tracking(env, "twist", std=0.06)[0] == pytest.approx(1.0)


def test_body_position_reward_uses_command_target_fixed_in_world():
  task = "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack"
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  term = env.command_manager.get_term("twist")
  robot = env.scene["robot"]
  params = env.reward_manager.get_term_cfg("body_position").params
  target0 = term.target_pos_w.clone()
  base0 = robot.data.root_link_pos_w[:, :2].clone()
  alive = torch.ones(4, dtype=torch.bool)  # envs not reset yet (a reset legitimately resamples the target)
  moved = torch.zeros(4)
  for _ in range(40):  # 0.8 s < minimum resampling time (4 s)
    _, _, terminated, truncated, _ = env.step(0.2 * (2 * torch.rand(4, NUM_JOINTS) - 1))
    alive &= ~(terminated | truncated)
    # Same target as the displacement command, and it does not follow the pelvis.
    assert torch.equal(term.target_pos_w[alive], target0[alive])
    err = (term.target_pos_w - robot.data.root_link_pos_w[:, :2]) @ term.direction
    expected = torch.exp(-(err / params["std"]) ** 2)
    assert torch.allclose(mdp.body_position_tracking(env, **params), expected)
    moved = torch.where(alive, (robot.data.root_link_pos_w[:, :2] - base0).norm(dim=-1), moved)
  assert alive.any() and moved[alive].max() > 0.005  # the body moved while its target stayed put
  env.close()


def test_postrack_differs_from_gain2p5_only_by_body_position_reward():
  from dataclasses import fields as dc_fields

  def same(x, y):  # repr without per-call function addresses
    return re.sub(r" at 0x[0-9a-f]+", "", repr(x)) == re.sub(r" at 0x[0-9a-f]+", "", repr(y))

  g = load_env_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-Gain2p5")
  p = load_env_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack")
  for f in dc_fields(g):
    if f.name != "rewards":
      assert same(getattr(g, f.name), getattr(p, f.name)), f.name
  assert set(p.rewards) - set(g.rewards) == {"body_position"} and set(g.rewards) <= set(p.rewards)
  for name in g.rewards:
    assert same(g.rewards[name], p.rewards[name]), name
  t = p.rewards["body_position"]
  assert t.func is mdp.body_position_tracking and t.weight == 1.0 and t.params == {"command_name": "twist", "std": 0.06}
  assert same(load_rl_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-Gain2p5"),
              load_rl_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack"))


def test_earlier_tasks_have_no_body_position_reward():
  for task in [t for t in list_tasks() if "Contact" in t and "PosTrack" not in t]:
    assert "body_position" not in load_env_cfg(task).rewards, task


# --- Stance (feet-midpoint) displacement reward (Move08-PosTrack-StanceTrack) ---

STANCE_TASK = "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack"


def _fake_stance_env(target_xy, feet_xyz, direction=(1.0, 0.0)):
  d = torch.tensor(direction)
  term = SimpleNamespace(target_pos_w=torch.tensor(target_xy, dtype=torch.float32), direction=d / d.norm())
  robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.tensor(feet_xyz, dtype=torch.float32)))
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: term), scene={"robot": robot})
  return env, SimpleNamespace(name="robot", site_ids=[0, 1])


def _stance(env, cfg, offset=0.014):
  return mdp.stance_position_tracking(env, "twist", std=0.06, offset=offset, asset_cfg=cfg)


def test_stance_reward_max_at_commanded_displacement_symmetric_monotone():
  # Pelvis target at the commanded displacement d; feet midpoint at d + offset + error.
  errs = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08]
  for d_cmd in (0.08, -0.08, 0.0):
    for sign in (1.0, -1.0):
      feet = [[[d_cmd + 0.014 + sign * e, 0.12, 0.0], [d_cmd + 0.014 + sign * e, -0.12, 0.0]] for e in errs]
      env, cfg = _fake_stance_env([[d_cmd, 0.0]] * len(errs), feet)
      r = _stance(env, cfg)
      assert r.shape == (len(errs),) and torch.all(torch.isfinite(r))
      assert r[0] == pytest.approx(1.0, abs=1e-6)
      assert torch.all(r[1:] < r[:-1])
      if sign > 0:
        pos = r
    assert torch.allclose(pos, r)  # symmetric in the sign of the error


def test_stance_reward_ignores_lateral_and_vertical_foot_motion():
  base = [[0.014, 0.12, 0.0], [0.014, -0.12, 0.0]]
  wide = [[0.014, 0.25, 0.0], [0.014, -0.25, 0.0]]       # stance width only
  shifted = [[0.014, 0.30, 0.0], [0.014, 0.05, 0.0]]     # both feet moved toward the wall
  lifted = [[0.014, 0.12, 0.08], [0.014, -0.12, 0.0]]    # swing foot in the air
  env, cfg = _fake_stance_env([[0.0, 0.0]] * 4, [base, wide, shifted, lifted])
  assert torch.allclose(_stance(env, cfg), torch.ones(4))


def test_stance_reward_uses_midpoint_including_swing_foot():
  # One foot steps 4 cm forward: midpoint moves 2 cm.
  env, cfg = _fake_stance_env([[0.02, 0.0]], [[[0.014 + 0.04, 0.12, 0.05], [0.014, -0.12, 0.0]]])
  assert _stance(env, cfg)[0] == pytest.approx(1.0, abs=1e-6)


def test_stance_offset_matches_reset_geometry_and_zero_command_is_max():
  cfg = load_env_cfg(STANCE_TASK)
  cfg.scene.num_envs = 4
  cfg.commands["twist"].displacement_range = 0.0  # zero command
  cfg.curriculum = {}  # the Move08 curriculum would otherwise set +-0.08 m at reset
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  robot = env.scene["robot"]
  params = env.reward_manager.get_term_cfg("stance_position").params
  sid = params["asset_cfg"].site_ids
  mid_x = robot.data.site_pos_w[:, sid, 0].mean(dim=1) - env.scene.env_origins[:, 0]
  pel_x = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  assert torch.allclose(mid_x - pel_x, torch.full_like(mid_x, params["offset"]), atol=1e-3)
  # Zero command: feet at their reset position give the maximum reward.
  assert torch.allclose(mdp.stance_position_tracking(env, **params), torch.ones(4), atol=1e-3)
  env.close()


def test_stance_target_is_fixed_while_feet_move():
  cfg = load_env_cfg(STANCE_TASK)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  term = env.command_manager.get_term("twist")
  robot = env.scene["robot"]
  params = env.reward_manager.get_term_cfg("stance_position").params
  sid = params["asset_cfg"].site_ids
  target0 = term.target_pos_w.clone()
  alive = torch.ones(4, dtype=torch.bool)
  for _ in range(40):
    _, _, terminated, truncated, _ = env.step(0.3 * (2 * torch.rand(4, NUM_JOINTS) - 1))
    alive &= ~(terminated | truncated)
    assert torch.equal(term.target_pos_w[alive], target0[alive])
    mid = robot.data.site_pos_w[:, sid, :2].mean(dim=1)
    err = (term.target_pos_w - mid) @ term.direction + params["offset"]
    assert torch.allclose(mdp.stance_position_tracking(env, **params), torch.exp(-(err / params["std"]) ** 2))
  # Deterministic displacement of the feet: teleport the robot 5 cm along the command axis.
  target1 = term.target_pos_w.clone()
  r0 = mdp.stance_position_tracking(env, **params)
  mid0 = robot.data.site_pos_w[:, sid, :2].mean(dim=1)
  pose = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w], dim=-1).clone()
  pose[:, :2] += 0.05 * term.direction
  robot.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  mid1 = robot.data.site_pos_w[:, sid, :2].mean(dim=1)
  assert torch.allclose((mid1 - mid0) @ term.direction, torch.full((4,), 0.05), atol=1e-4)
  assert torch.equal(term.target_pos_w, target1)  # the target does not follow the feet
  err = (target1 - mid1) @ term.direction + params["offset"]
  r1 = mdp.stance_position_tracking(env, **params)
  assert torch.allclose(r1, torch.exp(-(err / params["std"]) ** 2)) and not torch.allclose(r1, r0)
  env.close()


def test_stancetrack_differs_from_postrack_only_by_stance_reward():
  from dataclasses import fields as dc_fields

  def same(x, y):  # repr without per-call function addresses
    return re.sub(r" at 0x[0-9a-f]+", "", repr(x)) == re.sub(r" at 0x[0-9a-f]+", "", repr(y))

  p = load_env_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack")
  s = load_env_cfg(STANCE_TASK)
  for f in dc_fields(p):
    if f.name != "rewards":
      assert same(getattr(p, f.name), getattr(s, f.name)), f.name
  assert set(s.rewards) - set(p.rewards) == {"stance_position"} and set(p.rewards) <= set(s.rewards)
  for name in p.rewards:
    assert same(p.rewards[name], s.rewards[name]), name
  t = s.rewards["stance_position"]
  assert t.func is mdp.stance_position_tracking and t.weight == 1.0
  assert (t.params["std"], t.params["offset"], t.params["asset_cfg"].site_names) == (0.06, 0.014, ("left_foot", "right_foot"))
  assert same(load_rl_cfg("Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack"), load_rl_cfg(STANCE_TASK))


def test_earlier_tasks_have_no_stance_reward():
  for task in [t for t in list_tasks() if "Contact" in t and "StanceTrack" not in t]:
    assert "stance_position" not in load_env_cfg(task).rewards, task


# --- Per-foot commanded-displacement reward (Move08-PosTrack-StanceTrack-FootTrack) ---

FOOT_TASK = "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack-FootTrack"
REF = 0.014  # each foot's reset x relative to the pelvis target


def _feet_env(d, left_dx, right_dx, left_extra=(0.0, 0.0), right_extra=(0.0, 0.0)):
  """Pelvis target at displacement d; feet at reference + their displacement (+ optional y/z)."""
  feet = [[[REF + left_dx, 0.12 + left_extra[0], left_extra[1]], [REF + right_dx, -0.12 + right_extra[0], right_extra[1]]]]
  return _fake_stance_env([[d, 0.0]], feet)


def _foot(env, cfg, offsets=(REF, REF)):
  return mdp.foot_position_tracking(env, "twist", std=0.06, offsets=offsets, asset_cfg=cfg)[0].item()


def test_foot_reward_ordering_and_symmetry():
  for d in (0.08, -0.08):
    s = 1.0 if d > 0 else -1.0
    r = {k: _foot(*_feet_env(d, s * l, s * rr)) for k, (l, rr) in
         {"both": (0.08, 0.08), "trail4": (0.08, 0.04), "trail2": (0.08, 0.02), "leader": (0.08, 0.0),
          "wrong": (0.08, -0.02), "none": (0.0, 0.0), "both_wrong": (-0.02, -0.02)}.items()}
    assert r["both"] == pytest.approx(1.0, abs=1e-6)
    assert r["both"] > r["trail4"] > r["trail2"] > r["leader"] > r["wrong"] > r["none"]
    assert r["both_wrong"] < r["none"]  # moving both feet the wrong way is worse than not moving
    assert r["leader"] == pytest.approx(0.5 * (1 + math.exp(-(0.08 / 0.06) ** 2)), abs=1e-6)  # partial, not full
    if d > 0:
      pos = r
  assert all(pos[k] == pytest.approx(r[k], abs=1e-6) for k in r)  # symmetric in d


def test_foot_reward_keeps_each_foot_reference_and_zero_command():
  # Zero command: each foot at its own reference is maximal (no incentive to align or move feet).
  assert _foot(*_feet_env(0.0, 0.0, 0.0)) == pytest.approx(1.0, abs=1e-6)
  # Different per-foot references are honoured (target = own reference + d, not a shared point).
  env, cfg = _fake_stance_env([[0.05, 0.0]], [[[0.03 + 0.05, 0.12, 0.0], [-0.01 + 0.05, -0.12, 0.0]]])
  assert _foot(env, cfg, offsets=(0.03, -0.01)) == pytest.approx(1.0, abs=1e-6)
  assert _foot(env, cfg, offsets=(REF, REF)) < 0.95


def test_foot_reward_ignores_lateral_and_vertical_motion():
  base = _foot(*_feet_env(0.08, 0.08, 0.02))
  moved = _foot(*_feet_env(0.08, 0.08, 0.02, left_extra=(0.15, 0.06), right_extra=(-0.1, 0.03)))
  assert moved == pytest.approx(base, abs=1e-6)
  env, cfg = _feet_env(0.08, 0.08, 0.02)
  out = mdp.foot_position_tracking(env, "twist", std=0.06, offsets=(REF, REF), asset_cfg=cfg)
  assert out.shape == (1,) and torch.all(torch.isfinite(out))


def test_foot_offsets_match_reset_geometry():
  cfg = load_env_cfg(FOOT_TASK)
  cfg.scene.num_envs = 4
  cfg.commands["twist"].displacement_range = 0.0
  cfg.curriculum = {}
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  robot = env.scene["robot"]
  params = env.reward_manager.get_term_cfg("foot_position").params
  sid = params["asset_cfg"].site_ids
  rel = robot.data.site_pos_w[:, sid, 0] - robot.data.root_link_pos_w[:, None, 0]
  assert torch.allclose(rel, torch.tensor(params["offsets"]).expand_as(rel), atol=1e-3)
  assert torch.allclose(mdp.foot_position_tracking(env, **params), torch.ones(4), atol=1e-3)
  env.close()


def test_foot_targets_fixed_while_feet_move_and_follow_resampling():
  cfg = load_env_cfg(FOOT_TASK)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  term = env.command_manager.get_term("twist")
  robot = env.scene["robot"]
  params = env.reward_manager.get_term_cfg("foot_position").params
  sid = params["asset_cfg"].site_ids
  off = torch.tensor(params["offsets"])

  def expected():
    feet = robot.data.site_pos_w[:, sid, :2]
    e = ((term.target_pos_w[:, None, :] - feet) * term.direction).sum(-1) + off
    return torch.exp(-(e / params["std"]) ** 2).mean(-1)

  target0 = term.target_pos_w.clone()
  r0 = mdp.foot_position_tracking(env, **params)
  # Teleport the robot 5 cm along the axis: feet move, targets do not.
  pose = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w], dim=-1).clone()
  pose[:, :2] += 0.05 * term.direction
  robot.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  assert torch.equal(term.target_pos_w, target0)
  r1 = mdp.foot_position_tracking(env, **params)
  assert torch.allclose(r1, expected()) and not torch.allclose(r1, r0)
  # Resampling the command moves both foot targets by the new displacement.
  term.cfg.displacement_range = 0.08
  term._resample_command(torch.arange(4))
  assert not torch.equal(term.target_pos_w, target0)
  assert torch.allclose(mdp.foot_position_tracking(env, **params), expected())
  env.close()


def test_foottrack_differs_from_stancetrack_only_by_foot_reward():
  from dataclasses import fields as dc_fields

  def same(x, y):
    return re.sub(r" at 0x[0-9a-f]+", "", repr(x)) == re.sub(r" at 0x[0-9a-f]+", "", repr(y))

  s = load_env_cfg(STANCE_TASK)
  f = load_env_cfg(FOOT_TASK)
  for fld in dc_fields(s):
    if fld.name != "rewards":
      assert same(getattr(s, fld.name), getattr(f, fld.name)), fld.name
  assert set(f.rewards) - set(s.rewards) == {"foot_position"} and set(s.rewards) <= set(f.rewards)
  for name in s.rewards:
    assert same(s.rewards[name], f.rewards[name]), name
  t = f.rewards["foot_position"]
  assert t.func is mdp.foot_position_tracking and t.weight == 1.0
  assert (t.params["std"], t.params["offsets"], t.params["asset_cfg"].site_names) == (0.06, (0.014, 0.014), ("left_foot", "right_foot"))
  assert same(load_rl_cfg(STANCE_TASK), load_rl_cfg(FOOT_TASK))


def test_earlier_tasks_have_no_foot_reward():
  for task in [t for t in list_tasks() if "Contact" in t and "FootTrack" not in t]:
    assert "foot_position" not in load_env_cfg(task).rewards, task


ACTOR_CMD_TASK = FOOT_TASK + "-ActorCmd1p5"


def _actor_pair(gain_cls, n=8, dim=104, seed=0):
  from rsl_rl.models.mlp_model import MLPModel
  from tensordict import TensorDict

  g = torch.Generator().manual_seed(seed)
  obs = TensorDict({"actor": torch.randn(n, dim, generator=g) * 0.1}, batch_size=[n])
  kw = dict(obs_groups={"actor": ["actor"]}, obs_set="actor", output_dim=29, hidden_dims=(32, 16), activation="elu",
            obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})
  base = MLPModel(obs, **copy.deepcopy(kw)); amp = gain_cls(obs, **copy.deepcopy(kw))
  amp.load_state_dict(base.state_dict())
  return base, amp, obs


def test_amplify_command_scales_xy_clips_norm_and_keeps_other_dims():
  from src.tasks.stable_contact.actor_command_gain import COMMAND_XY, MAX_SPEED, amplify_command

  assert COMMAND_XY == (6, 8) and MAX_SPEED == 0.2
  x = torch.randn(5, 104)
  x[:, 6:8] = torch.tensor([[0.05, 0.0], [0.0, -0.04], [0.03, 0.04], [0.15, 0.0], [-0.12, 0.09]])
  before = x.clone()
  y = amplify_command(x, 1.5)
  assert torch.equal(x, before)  # input not mutated
  assert torch.allclose(y[:3, 6:8], 1.5 * x[:3, 6:8])
  assert torch.allclose(torch.linalg.norm(y[3:, 6:8], dim=-1), torch.full((2,), 0.2))  # clipped to max_speed, direction kept
  assert torch.allclose(y[3:, 6:8] / torch.linalg.norm(y[3:, 6:8], dim=-1, keepdim=True), x[3:, 6:8] / torch.linalg.norm(x[3:, 6:8], dim=-1, keepdim=True))
  keep = [i for i in range(104) if i not in (6, 7)]
  assert torch.equal(y[:, keep], x[:, keep])  # wz (dim 8), phase and everything else untouched
  assert torch.allclose(amplify_command(x, 1.0)[:3], x[:3])


def test_actor_cmd_gain_normalizer_tracks_real_obs_and_latent_uses_amplified_command():
  from src.tasks.stable_contact.actor_command_gain import ActorCommandGain1p5, amplify_command

  base, amp, obs = _actor_pair(ActorCommandGain1p5)
  base.train(); amp.train()
  for _ in range(3):
    base.update_normalization(obs); amp.update_normalization(obs)
  # identical statistics: the normalizer learns from the real (unamplified) observation
  for k in ("_mean", "_var", "_std", "count"):
    assert torch.equal(getattr(base.obs_normalizer, k), getattr(amp.obs_normalizer, k)), k
  with torch.no_grad():
    lat = amp.get_latent(obs)
    assert torch.allclose(lat, base.obs_normalizer(amplify_command(obs["actor"], 1.5)))
    d = lat - base.get_latent(obs)
    assert torch.count_nonzero(d[:, [i for i in range(104) if i not in (6, 7)]]) == 0
    # with the same statistics the extra input is exactly (gain - 1) * v / (std + eps) in the unclipped range
    v = obs["actor"][:, 6:8]; s = base.obs_normalizer._std[0, 6:8] + base.obs_normalizer.eps
    free = torch.linalg.norm(1.5 * v, dim=-1) < 0.2
    assert free.sum() >= 2 and (~free).sum() >= 1
    assert torch.allclose(d[free, 6:8], 0.5 * v[free] / s, atol=1e-5)


def test_actor_cmd_gain_export_applies_same_transform():
  from src.tasks.stable_contact.actor_command_gain import ActorCommandGain1p5

  _, amp, obs = _actor_pair(ActorCommandGain1p5)
  amp.train(); amp.update_normalization(obs); amp.eval()
  with torch.no_grad():
    ref = amp(obs)
    assert torch.allclose(amp.as_onnx(verbose=False)(obs["actor"]), ref, atol=1e-6)
    assert torch.allclose(amp.as_jit()(obs["actor"]), ref, atol=1e-6)
    assert amp.as_onnx(verbose=False).get_dummy_inputs()[0].shape == (1, 104)


def test_actor_command_slice_matches_movement_actor_observation_layout():
  cfg = load_env_cfg(ACTOR_CMD_TASK); cfg.scene.num_envs = 1
  e = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  names = e.observation_manager.active_terms["actor"]; dims = e.observation_manager.group_obs_term_dim["actor"]
  i = names.index("command")
  assert sum(int(np.prod(d)) for d in dims[:i]) == 6 and int(np.prod(dims[i])) == 3
  assert sum(int(np.prod(d)) for d in dims) == 104
  assert e.command_manager.get_term("twist").cfg.max_speed == 0.2
  e.close()


def test_actorcmd1p5_differs_from_foottrack_only_in_actor_class():
  from dataclasses import fields as dc_fields

  def same(x, y):
    return re.sub(r" at 0x[0-9a-f]+", "", repr(x)) == re.sub(r" at 0x[0-9a-f]+", "", repr(y))

  f, a = load_env_cfg(FOOT_TASK), load_env_cfg(ACTOR_CMD_TASK)
  for fld in dc_fields(f):
    assert same(getattr(f, fld.name), getattr(a, fld.name)), fld.name
  rf, ra = load_rl_cfg(FOOT_TASK), load_rl_cfg(ACTOR_CMD_TASK)
  assert ra.actor.class_name == "src.tasks.stable_contact.actor_command_gain:ActorCommandGain1p5"
  assert rf.actor.class_name == "MLPModel" and ra.critic.class_name == "MLPModel"
  ra.actor.class_name = rf.actor.class_name
  assert same(rf, ra)


def test_earlier_tasks_use_plain_actor():
  for task in [t for t in list_tasks() if "Contact" in t and "ActorCmd" not in t]:
    assert load_rl_cfg(task).actor.class_name == "MLPModel", task

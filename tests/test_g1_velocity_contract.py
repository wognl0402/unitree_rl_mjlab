"""CPU checks for the Unitree G1 velocity controller interface.

These pin the contract between training (src/tasks/velocity), the exported
policy, and the hand-written deploy config used by deploy/robots/g1. Building
the environment uses mujoco_warp on the CPU device, so no GPU is needed.

Run: python -m pytest tests -q
"""

import math
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import torch
import yaml

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from src.assets.robots import G1_ACTION_SCALE
from src.assets.robots.unitree_g1 import g1_constants
from src.tasks.velocity import mdp

REPO = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO / "deploy/robots/g1/config/policy/velocity/v0"
TASK = "Unitree-G1-Flat"

# Training observation term -> name registered in deploy/include/.../observations.h.
# The C++ side concatenates terms in deploy.yaml order, so order must match too.
TRAIN_TO_DEPLOY_OBS = {
  "base_ang_vel": "base_ang_vel",
  "projected_gravity": "projected_gravity",
  "command": "velocity_commands",
  "phase": "gait_phase",
  "joint_pos": "joint_pos_rel",
  "joint_vel": "joint_vel_rel",
  "actions": "last_action",
}
NUM_JOINTS = 29


@pytest.fixture(scope="module")
def deploy_cfg() -> dict:
  with open(DEPLOY_DIR / "params/deploy.yaml") as f:
    return yaml.safe_load(f)


@pytest.fixture(scope="module")
def shipped_policy_metadata() -> dict[str, str]:
  model = onnx.load(DEPLOY_DIR / "exported/policy.onnx", load_external_data=False)
  return {p.key: p.value for p in model.metadata_props}


@pytest.fixture(scope="module")
def env():
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  yield env
  env.close()


##
# Config-only checks (no simulation).
##


def test_g1_tasks_registered():
  tasks = set(list_tasks())
  for task in ("Unitree-G1-Flat", "Unitree-G1-Rough", "Unitree-G1-23Dof-Flat"):
    assert task in tasks


def test_action_scale_is_quarter_effort_over_stiffness():
  for actuator in g1_constants.G1_ARTICULATION.actuators:
    for expr in actuator.target_names_expr:
      expected = 0.25 * actuator.effort_limit / actuator.stiffness
      assert G1_ACTION_SCALE[expr] == pytest.approx(expected)


def test_actor_observation_order_matches_deploy(deploy_cfg):
  cfg = load_env_cfg(TASK)
  train_terms = list(cfg.observations["actor"].terms)
  assert [TRAIN_TO_DEPLOY_OBS[t] for t in train_terms] == list(deploy_cfg["observations"])


def test_gait_period_matches_between_obs_reward_and_deploy(deploy_cfg):
  cfg = load_env_cfg(TASK)
  obs_period = cfg.observations["actor"].terms["phase"].params["period"]
  assert cfg.rewards["foot_gait"].params["period"] == obs_period
  assert deploy_cfg["observations"]["gait_phase"]["params"]["period"] == obs_period


def test_deploy_command_ranges_within_training_ranges(deploy_cfg):
  cfg = load_env_cfg(TASK)
  train = cfg.commands["twist"].ranges
  deploy_ranges = deploy_cfg["commands"]["base_velocity"]["ranges"]
  for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
    lo, hi = deploy_ranges[axis]
    train_lo, train_hi = getattr(train, axis)
    assert train_lo <= lo and hi <= train_hi, axis


##
# Pure MDP term checks with a stub env.
##


def _stub_env(episode_steps, command, contact_time=None):
  n = len(episode_steps)
  sensor = SimpleNamespace(data=SimpleNamespace(current_contact_time=contact_time))
  return SimpleNamespace(
    num_envs=n,
    device="cpu",
    step_dt=0.02,
    episode_length_buf=torch.tensor(episode_steps),
    command_manager=SimpleNamespace(get_command=lambda name: command),
    scene={"feet_ground_contact": sensor},
  )


def test_phase_is_unit_circle_and_zero_when_standing():
  command = torch.tensor([[0.5, 0.0, 0.0], [0.05, 0.0, 0.0]])
  env = _stub_env([15, 15], command)
  out = mdp.phase(env, period=0.6, command_name="twist")
  # 15 steps * 0.02 s = 0.3 s = half a period.
  assert out[0].tolist() == pytest.approx([math.sin(math.pi), math.cos(math.pi)], abs=1e-6)
  assert out[1].tolist() == [0.0, 0.0]


def test_feet_gait_rewards_alternating_contacts():
  command = torch.tensor([[1.0, 0.0, 0.0]] * 3)
  # Step 0: left leg phase 0.0 (stance), right leg phase 0.5 (stance, < 0.56).
  # Step 18: left phase 0.6 (swing), right phase 0.1 (stance).
  contact = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
  env = _stub_env([18, 18, 0], command, contact_time=contact)
  out = mdp.feet_gait(
    env, period=0.6, offset=[0.0, 0.5], threshold=0.56, command_threshold=0.1,
    command_name="twist", sensor_name="feet_ground_contact",
  )
  assert out.tolist() == pytest.approx([0.0, 1.0, 1.0])


def test_feet_gait_disabled_below_command_threshold():
  command = torch.tensor([[0.05, 0.0, 0.0]])
  env = _stub_env([0], command, contact_time=torch.tensor([[1.0, 1.0]]))
  out = mdp.feet_gait(
    env, period=0.6, offset=[0.0, 0.5], threshold=0.56, command_threshold=0.1,
    command_name="twist", sensor_name="feet_ground_contact",
  )
  assert out.tolist() == [0.0]


##
# Built environment checks (1 env, CPU).
##


def test_dimensions(env):
  assert env.action_manager.total_action_dim == NUM_JOINTS
  om = env.observation_manager
  assert om.group_obs_dim["actor"] == (3 + 3 + 3 + 2 + 3 * NUM_JOINTS,)
  assert env.step_dt == pytest.approx(0.02)
  assert env.physics_dt == pytest.approx(0.005)


def test_joint_order_matches_shipped_policy(env, shipped_policy_metadata, deploy_cfg):
  robot = env.scene["robot"]
  action = env.action_manager.get_term("joint_pos")
  deploy_joint_names = shipped_policy_metadata["joint_names"].split(",")
  assert list(robot.joint_names) == deploy_joint_names
  assert list(action.target_names) == deploy_joint_names
  assert deploy_cfg["joint_ids_map"] == list(range(NUM_JOINTS))


def test_gains_scale_and_default_pose_match_deploy(env, deploy_cfg):
  robot = env.scene["robot"]
  action = env.action_manager.get_term("joint_pos")
  model = env.sim.mj_model
  ctrl_id = {model.actuator(i).name.split("/")[-1]: i for i in range(model.nu)}
  ids = [ctrl_id[name] for name in robot.joint_names]
  kp = model.actuator_gainprm[ids, 0].tolist()
  kd = (-model.actuator_biasprm[ids, 2]).tolist()
  # deploy.yaml stores values rounded to 1-2 decimals.
  assert kp == pytest.approx(deploy_cfg["stiffness"], abs=0.06)
  assert kd == pytest.approx(deploy_cfg["damping"], abs=0.06)
  joint_action = deploy_cfg["actions"]["JointPositionAction"]
  assert action.scale[0].tolist() == pytest.approx(joint_action["scale"], abs=0.006)
  default = robot.data.default_joint_pos[0].tolist()
  assert default == pytest.approx(deploy_cfg["default_joint_pos"], abs=1e-6)
  assert default == pytest.approx(joint_action["offset"], abs=1e-6)


def test_action_maps_to_default_plus_scaled_target(env):
  robot = env.scene["robot"]
  action_term = env.action_manager.get_term("joint_pos")
  raw = torch.linspace(-1.0, 1.0, NUM_JOINTS).unsqueeze(0)
  env.action_manager.process_action(raw)
  expected = robot.data.default_joint_pos + raw * action_term.scale
  assert torch.allclose(action_term._processed_actions, expected)


def test_zero_action_rollout_is_finite(env):
  env.reset()
  zero = torch.zeros(1, NUM_JOINTS)
  for _ in range(25):
    obs, reward, _, _, _ = env.step(zero)
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(obs["critic"]).all()
    assert torch.isfinite(reward).all()

"""CPU tests for the stable-contact primitives and the StableContactStand harness.

Pure-math tests use stub tensors. Simulation tests drive the development
harness with the scripted probe from scripts/diagnose_hand_contact.py (root
held, arm moved by IK) and check the measurements against MuJoCo itself.

Run: python -m pytest tests/test_stable_contact.py -q
"""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
import src.tasks.stable_contact.mdp as mdp
from mjlab.envs import ManagerBasedRlEnv
from src.assets.props.wall import WALL_OUTWARD_NORMAL_LOCAL
from src.tasks.stable_contact.stable_contact_env_cfg import (
  G1_HANDS,
  SURFACE,
  contact_target_local,
  hand_sensor_name,
  make_contact_reward_terms,
  make_stable_contact_stand_env_cfg,
)

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
  "diagnose_hand_contact", REPO / "scripts/diagnose_hand_contact.py"
)
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
N_LEFT_WALL = torch.tensor([[0.0, -1.0, 0.0]])  # Outward normal of a wall at +y.


##
# Pure math.
##


@pytest.mark.parametrize("device", DEVICES)
def test_force_decomposition(device):
  n = N_LEFT_WALL.to(device)
  # Hand pushes into the wall (+y) with 10 N and drags 3 N along +x, 4 N along -z.
  force = torch.tensor([[3.0, 10.0, -4.0]], device=device)
  f_n = mdp.normal_force(force, n)
  f_t = mdp.tangential_component(force, n)
  assert f_n.device == force.device and f_t.device == force.device
  assert f_n.item() == pytest.approx(10.0)
  assert f_t.cpu().tolist()[0] == pytest.approx([3.0, 0.0, -4.0])
  assert torch.sum(f_t * n).item() == pytest.approx(0.0, abs=1e-6)
  # Pulling away from the wall gives a negative pressing force.
  assert mdp.normal_force(-force, n).item() == pytest.approx(-10.0)


@pytest.mark.parametrize("device", DEVICES)
def test_tangential_velocity_grows_with_lateral_speed(device):
  n = N_LEFT_WALL.to(device)
  speeds = []
  for lateral in (0.0, 0.1, 0.5):
    v = torch.tensor([[lateral, 0.3, 0.0]], device=device)  # 0.3 m/s into the wall.
    speeds.append(torch.linalg.norm(mdp.tangential_component(v, n)).item())
  assert speeds[0] == pytest.approx(0.0, abs=1e-7)
  assert speeds[0] < speeds[1] < speeds[2]
  assert speeds[2] == pytest.approx(0.5)


def test_friction_cone_ratio():
  f_n = torch.tensor([10.0, 10.0, 0.0])
  f_t = torch.tensor([5.0, 10.0, 3.0])
  ratio = mdp.friction_cone_ratio(f_n, f_t, mu=1.0)
  assert ratio.tolist() == pytest.approx([0.5, 1.0, 0.0])


##
# Reward terms on a stub env (one wall at +y, hand sensor, palm velocity).
##


def _stub_env(force, found, palm_vel, command=None):
  b = force.shape[0]
  yaw = math.pi / 2  # Wall local +x -> world +y.
  wall = SimpleNamespace(data=SimpleNamespace(
    root_link_quat_w=torch.tensor([[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]]).expand(b, 4),
    root_link_pos_w=torch.tensor([[0.0, 0.4, 1.0]]).expand(b, 3),
  ))
  sensor = SimpleNamespace(
    cfg=SimpleNamespace(reduce="netforce"),
    data=SimpleNamespace(found=found.view(b, 1), force=force.view(b, 1, 3)),
  )
  robot = SimpleNamespace(data=SimpleNamespace(site_lin_vel_w=palm_vel.view(b, 1, 3)))
  return SimpleNamespace(
    num_envs=b,
    scene={SURFACE: wall, "hand": sensor, "robot": robot},
    command_manager=SimpleNamespace(get_command=lambda name: command),
  )


SURF = {"surface_name": SURFACE, "normal_local": WALL_OUTWARD_NORMAL_LOCAL}
SITE = SimpleNamespace(name="robot", site_ids=[0])


def test_force_band_is_bounded_and_needs_contact():
  pressing = torch.tensor([0.0, 5.0, 15.0, 1000.0, 15.0])
  force = torch.stack([torch.zeros(5), pressing, torch.zeros(5)], dim=-1)
  found = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
  env = _stub_env(force, found, torch.zeros(5, 3))
  r = mdp.hand_contact_force_band(env, "hand", **SURF, force_range=(10.0, 20.0), std=5.0)
  assert r[2].item() == pytest.approx(1.0)  # Inside the band.
  assert r[1].item() == pytest.approx(math.exp(-1.0))  # 5 N below the band.
  assert r[3].item() == pytest.approx(0.0, abs=1e-6)  # Smashing earns nothing.
  assert r[4].item() == 0.0  # No contact -> no reward.
  assert torch.all(r <= 1.0)
  excess = mdp.hand_contact_force_excess(env, "hand", **SURF, force_limit=20.0)
  assert excess[:3].tolist() == [0.0, 0.0, 0.0]
  assert excess[3].item() == pytest.approx(((1000.0 - 20.0) / 20.0) ** 2)


def test_slip_counts_only_uncommanded_in_plane_motion():
  found = torch.ones(3)
  palm_vel = torch.tensor([
    [0.0, 0.0, 0.0],  # Stationary.
    [0.2, 0.0, 0.0],  # Sliding forward along the wall.
    [0.0, 0.3, 0.0],  # Moving into the wall: not slip.
  ])
  env = _stub_env(torch.zeros(3, 3), found, palm_vel)
  cost = mdp.hand_slip(env, "hand", **SURF, asset_cfg=SITE)
  assert cost.tolist() == pytest.approx([0.0, 0.04, 0.0])
  # Commanding the same forward sliding velocity removes the penalty.
  command = torch.tensor([[0.2, 0.0, 0.0]]).expand(3, 3)
  env = _stub_env(torch.zeros(3, 3), found, palm_vel, command=command)
  cost = mdp.hand_slip(env, "hand", **SURF, asset_cfg=SITE, command_name="contact")
  assert cost.tolist() == pytest.approx([0.04, 0.0, 0.04])


def test_slip_is_zero_without_contact():
  env = _stub_env(torch.zeros(1, 3), torch.zeros(1), torch.tensor([[0.5, 0.0, 0.0]]))
  assert mdp.hand_slip(env, "hand", **SURF, asset_cfg=SITE).item() == 0.0


##
# Config (no simulation).
##


def test_unknown_active_hand_rejected():
  with pytest.raises(ValueError):
    make_stable_contact_stand_env_cfg(active_hand="both")


def test_target_is_mirrored_per_hand():
  left = contact_target_local("left", forward=0.15, height_offset=0.0)
  right = contact_target_local("right", forward=0.15, height_offset=0.0)
  assert left[1] == -right[1] and left[0] == right[0] < 0


def test_baseline_velocity_task_untouched():
  from mjlab.tasks.registry import list_tasks, load_env_cfg

  # Only the separately registered experimental tasks; the baseline is unchanged.
  assert sorted(t for t in list_tasks() if "Contact" in t) == [
    "Unitree-G1-StableContactMove", "Unitree-G1-StableContactMove-GravHold",
    "Unitree-G1-StableContactMove-GravHold-Contact4", "Unitree-G1-StableContactMove-GravHold-ForceBand",
    "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08",
    "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-Gain2p5",
    "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack",
    "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack",
    "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack-FootTrack",
    "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack-FootTrack-ActorCmd1p5",
    "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack-FootTrack-ActorCmd1p5-FixedStd",
    "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack-FootTrack-ActorCmd1p5-StanceRefP025",
    "Unitree-G1-StableContactMove-GravHold-ForceBand-Move08-PosTrack-StanceTrack-FootTrack-ActorCmd1p5-StanceRefP025-FixedStd",
    "Unitree-G1-StableContactMove-GravHold-Slip1"]
  cfg = load_env_cfg("Unitree-G1-Flat")
  assert "wall" not in cfg.scene.entities
  assert cfg.commands["twist"].rel_standing_envs == pytest.approx(0.05)


##
# Built harness (2 envs, no scripted motion).
##


@pytest.fixture(scope="module")
def harness():
  cfg = make_stable_contact_stand_env_cfg(active_hand="left")
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  yield env
  env.close()


def test_harness_shapes_and_finite(harness):
  obs, *_ = harness.step(torch.zeros(2, 29))
  assert obs["actor"].shape == (2, 98 + 6)
  assert obs["critic"].shape == (2, 113 + 6 + 5)
  assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
  om = harness.observation_manager
  dims = dict(zip(om.active_terms["critic"], om.group_obs_term_dim["critic"]))
  assert dims["hand_to_target"] == (3,) and dims["surface_normal"] == (3,)
  assert dims["hand_contact"] == (1,) and dims["hand_force"] == (2,)
  assert dims["hand_slip"] == (1,) and dims["hand_contact_time"] == (1,)
  for name in ("hand_contact", "hand_force", "hand_slip", "hand_contact_time"):
    assert name not in om.active_terms["actor"], f"privileged term {name} in actor"


def test_harness_geometry(harness):
  # Base-frame normal also reflects the robot's small postural drift after a step.
  normal_b = mdp.surface_normal_b(harness, **SURF)
  assert torch.allclose(normal_b, torch.tensor([[0.0, -1.0, 0.0]] * 2), atol=1e-2)
  wall = harness.scene[SURFACE].data.root_link_pos_w - harness.scene.env_origins
  assert torch.allclose(wall, torch.tensor([[0.0, 0.4, 1.0]] * 2), atol=1e-6)


def test_posture_terms_skip_active_arm_only(harness):
  robot = harness.scene["robot"]
  ids = harness.reward_manager.get_term_cfg("pose").params["asset_cfg"].joint_ids
  names = {robot.joint_names[i] for i in ids}
  assert len(names) == 22
  assert not any(n.startswith(("left_shoulder", "left_elbow", "left_wrist")) for n in names)
  assert "right_elbow_joint" in names and "left_knee_joint" in names


def test_all_contact_rewards_build_and_stay_finite():
  # Test-only weights/thresholds: they exercise the plumbing, not a tuned task.
  weights = dict.fromkeys(
    ["hand_contact", "hand_force_band", "hand_force_excess", "hand_slip",
     "hand_friction_cone", "hand_target", "hand_contact_persistence",
     "hand_contact_break", "unintended_contact"], 1.0)
  terms = make_contact_reward_terms(
    "right", weights, force_range=(5.0, 20.0), force_std=5.0, force_limit=40.0,
    target_local=contact_target_local("right", 0.15, 0.0), target_std=0.1, cone_margin=0.8,
    persistence_time=1.0)
  cfg = make_stable_contact_stand_env_cfg(active_hand="right", contact_rewards=terms)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  for _ in range(5):
    _, reward, *_ = env.step(torch.zeros(2, 29))
    assert torch.isfinite(reward).all()
  assert set(weights) <= set(env.reward_manager.active_terms)
  env.close()
  with pytest.raises(ValueError):
    make_contact_reward_terms("left", {"hand_smash": 1.0}, force_range=(1, 2), force_std=1,
                              force_limit=3, target_local=(0, 0, 0), target_std=1,
                              cone_margin=1, persistence_time=1)


##
# Scripted contact scenarios (root held, arm driven by IK).
##


def _run(probe, palm_target_w, steps):
  probe.set_palm_target(np.asarray(palm_target_w))
  out = []
  for _ in range(steps):
    probe.step()
    out.append(probe.measure())
  return out


BASE_PUSH = 5.0  # N, holds the hand on the wall.
PUSHES = (0.0, 10.0, 20.0)  # N, extra load on top of BASE_PUSH.


def _scenario(hand):
  probe = diag.ContactProbe(hand)
  fwd = G1_HANDS[hand].forward_sign
  res = {"free": _run(probe, probe.wall_point_w((-0.20, fwd * 0.15, 0.0)), 50)}
  feet = probe.env.scene["feet_ground_contact"].data.found
  res["feet_on_ground"] = bool((feet > 0).any())
  # Light press (palm target 1 cm past the contact standoff) held by a known
  # base push, then known extra pushes. Open-loop pressing alone is marginal
  # (it chatters or drifts toward torque saturation from run to run); the base
  # push makes contact firm. A deeper joint target is not a controlled "harder
  # press": the resulting force is path dependent.
  press = probe.wall_point_w((diag.LIGHT_PRESS_DEPTH, fwd * 0.15, -0.15))
  for extra in PUSHES:
    probe.push_into_wall(BASE_PUSH + extra)
    res[extra] = _run(probe, press, 60 if extra == 0.0 else 40)
  res["contact_time_before_reset"] = probe.measure().contact_time
  probe.push_into_wall(0.0)
  probe.env.reset()
  sensor = probe.env.scene[hand_sensor_name(hand)]
  res["contact_time_after_reset"] = float(sensor.data.current_contact_time[0, 0])
  res["air_time_after_reset"] = float(sensor.data.current_air_time[0, 0])
  probe.close()
  return res


@pytest.fixture(scope="module")
def left_scenario():
  return _scenario("left")


@pytest.fixture(scope="module")
def right_scenario():
  return _scenario("right")


def _mean(ms, f, tail=20):
  return float(np.mean([f(m) for m in ms[-tail:]]))


@pytest.mark.parametrize("scenario", ["left_scenario", "right_scenario"])
def test_no_contact_when_hand_is_off_the_wall(scenario, request):
  res = request.getfixturevalue(scenario)
  for m in res["free"][-20:]:
    assert not m.contact
    assert m.normal_force == pytest.approx(0.0, abs=1e-6)
    assert m.tangential_force == pytest.approx(0.0, abs=1e-6)
    assert m.slip_speed == 0.0
  # Wrong body: the feet are on the ground, yet nothing counts as hand contact.
  assert res["feet_on_ground"]


@pytest.mark.parametrize("scenario", ["left_scenario", "right_scenario"])
def test_pressing_harder_increases_normal_force(scenario, request):
  """Calibration: an extra push of P newtons into the wall raises the measured
  pressing force monotonically, by a clearly positive amount that never
  exceeds the push. The push acts at the hand's center of mass and the PD-held
  arm readjusts its own share, so the gain is configuration dependent: observed
  0.19-1.07 P across (non-bit-reproducible) Warp CPU runs, lowest when the arm
  already presses near actuator saturation. Absolute force magnitude is checked
  against native MuJoCo in test_sensor_force_matches_native_mujoco."""
  res = request.getfixturevalue(scenario)

  def pressing(p):
    window = [m for m in res[p][-15:] if m.contact]
    assert len(window) >= 12, "the pushed hand should stay in contact"
    return float(np.mean([m.normal_force for m in window]))

  base, f10, f20 = pressing(0.0), pressing(10.0), pressing(20.0)
  assert base > 0.0
  assert base < f10 < f20
  assert 0.1 * 10.0 <= f10 - base <= 1.5 * 10.0
  assert 0.1 * 20.0 <= f20 - base <= 1.5 * 20.0
  for p in (0.0, 10.0, 20.0):
    assert all(np.isfinite([m.normal_force, m.tangential_force, m.slip_speed]).all()
               for m in res[p])


@pytest.mark.parametrize("scenario", ["left_scenario", "right_scenario"])
def test_measured_normal_matches_mujoco(scenario, request):
  res = request.getfixturevalue(scenario)
  m = next(m for m in reversed(res[20.0]) if m.native_contact_normal is not None)
  hand = "left" if scenario.startswith("left") else "right"
  outward_y = -1.0 if hand == "left" else 1.0
  # MuJoCo's contact normal points hand -> wall, i.e. against the outward normal.
  assert m.native_contact_normal.tolist() == pytest.approx([0.0, -outward_y, 0.0], abs=1e-3)


@pytest.mark.parametrize("scenario", ["left_scenario", "right_scenario"])
def test_only_the_active_hand_counts(scenario, request):
  res = request.getfixturevalue(scenario)
  for p in (0.0, 10.0, 20.0):
    assert sum(m.contact for m in res[p][-15:]) >= 8
    assert not any(m.other_hand_contact for m in res[p])
    assert all(m.unintended_contacts == 0 for m in res[p])


@pytest.mark.parametrize("scenario", ["left_scenario", "right_scenario"])
def test_contact_time_persists_then_resets(scenario, request):
  res = request.getfixturevalue(scenario)
  assert res["contact_time_before_reset"] > 0.3
  assert res["contact_time_after_reset"] == 0.0
  assert res["air_time_after_reset"] == 0.0


def test_sensor_force_matches_native_mujoco():
  """At every pressing step, the sensor's pressing force matches the normal force
  native MuJoCo computes for the same (warm-started) state.

  Tolerance: mujoco_warp's 10-iteration solve differs from native MuJoCo by an
  offset that is constant within a run but varies between runs (observed
  0.3-6%), because Warp CPU runs are not bit-reproducible.
  """
  probe = diag.ContactProbe("left")
  probe.set_palm_target(probe.wall_point_w((diag.LIGHT_PRESS_DEPTH, -0.15, -0.15)))
  probe.push_into_wall(10.0)
  pairs = []
  for _ in range(60):
    probe.step()
    m = probe.measure()
    probe._sync_native()
    mujoco.mj_forward(probe.mj_model, probe.mj_data)
    d = probe.mj_data
    native = 0.0
    for i in range(d.ncon):
      c = d.contact[i]
      if {c.geom1, c.geom2} == {probe.hand_geom_id, probe.wall_geom_id}:
        f = np.zeros(6)
        mujoco.mj_contactForce(probe.mj_model, d, i, f)
        native += f[0]
    pairs.append((m.contact, m.normal_force, native))
  probe.close()
  in_contact = [(s, n) for c, s, n in pairs[20:] if c]
  assert len(in_contact) >= 30
  for sensor_fn, native_fn in in_contact:
    assert sensor_fn == pytest.approx(native_fn, rel=0.08, abs=0.05)


def test_wrong_surface_is_not_target_contact():
  probe = diag.ContactProbe("left")
  _run(probe, (0.08, 0.05, 0.95), 40)  # Left palm against the robot's own torso.
  m = probe.measure()
  probe._sync_native()
  mujoco.mj_forward(probe.mj_model, probe.mj_data)
  d = probe.mj_data
  touching = {
    probe.mj_model.geom(c.geom2 if c.geom1 == probe.hand_geom_id else c.geom1).name
    for c in d.contact[: d.ncon] if probe.hand_geom_id in (c.geom1, c.geom2)
  }
  probe.close()
  assert touching, "scenario should put the hand on some non-wall geom"
  assert "wall/wall_geom" not in touching
  assert not m.contact and m.normal_force == 0.0


def test_contact_break_and_persistence_follow_press_release_cycles():
  """Hold / release / hold: every release fires the break term and clears the
  contact timer and persistence reward; holds accumulate contact time.

  The scripted (open-loop) arm occasionally chatters even while holding, so
  holds are only required to build up persistence, not to be break-free.
  """
  probe = diag.ContactProbe("left")
  sensor = hand_sensor_name("left")
  press = probe.wall_point_w((0.04, -0.15, -0.15))
  release = probe.wall_point_w((-0.15, -0.15, -0.15))
  log = []  # (break flag, contact flag, contact time, persistence reward)
  for target, steps in [(press, 50), (release, 30), (press, 50), (release, 30)]:
    probe.set_palm_target(target)
    for _ in range(steps):
      probe.step()
      m = probe.measure()
      log.append((
        mdp.hand_contact_break(probe.env, sensor).item(), m.contact, m.contact_time,
        mdp.hand_contact_persistence(probe.env, sensor, saturation_time=1.0).item(),
      ))
  probe.close()
  dt = 0.02  # Policy step.
  for hold in (log[0:50], log[80:130]):
    timer = [t for _, _, t, _ in hold]
    ticks = sum(abs((b - a) - dt) < 1e-3 for a, b in zip(timer, timer[1:]))
    assert ticks >= 5, "contact timer should advance one step at a time while held"
    for _, _, t, r in hold:
      assert r == pytest.approx(min(t, 1.0), abs=1e-6)
  released_from_contact = 0
  for start in (49, 129):
    rel = log[start + 1:start + 31]
    if log[start][1]:  # In contact when the release began: it must register a break.
      released_from_contact += 1
      # Include the last hold step: the timers lag the end-of-step contact flag
      # by one physics substep, so the loss can be flagged on that step.
      assert sum(b for b, *_ in log[start:start + 31]) >= 1
    assert rel[-1][2] == 0.0 and rel[-1][3] == 0.0
  assert released_from_contact >= 1, "at least one hold should end in contact"


def test_persistence_reward_is_bounded_and_tracks_timer():
  timer = torch.tensor([[0.0], [0.25], [1.0], [7.0]])
  sensor = SimpleNamespace(data=SimpleNamespace(current_contact_time=timer))
  env = SimpleNamespace(scene={"hand": sensor})
  r = mdp.hand_contact_persistence(env, "hand", saturation_time=0.5)
  assert r.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])

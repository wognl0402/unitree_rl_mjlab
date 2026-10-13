"""CPU diagnostic: are hand/wall contact measurements physically meaningful?

DIAGNOSTIC ONLY. This script holds the robot root still and drives one arm with
inverse kinematics; none of that state manipulation is part of the RL task.
It builds the StableContactStand development config, so the sensors and mdp
terms measured here are exactly the ones a policy would be trained with.

Run: python scripts/diagnose_hand_contact.py --hand left
"""

import argparse
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
import src.tasks.stable_contact.mdp as mdp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.props.wall import WALL_GEOM_NAME, WALL_OUTWARD_NORMAL_LOCAL
from src.tasks.stable_contact.stable_contact_env_cfg import (
  G1_HANDS,
  HAND_CAPSULE_RADIUS,
  SURFACE,
  UNINTENDED_SENSOR,
  hand_sensor_name,
  make_stable_contact_stand_env_cfg,
)

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")
# Palm-site target 1 cm past the contact standoff (wall-frame depth). Deeper
# targets drive the arm toward torque saturation, where the measured response
# to an external push becomes run-dependent.
LIGHT_PRESS_DEPTH = -HAND_CAPSULE_RADIUS + 0.01


@dataclass
class Measurement:
  palm_w: np.ndarray
  palm_vel_w: np.ndarray
  contact: bool
  normal_force: float
  tangential_force: float
  friction_cone_ratio: float
  slip_speed: float
  contact_time: float
  other_hand_contact: bool
  unintended_contacts: int
  native_contact_normal: np.ndarray | None  # MuJoCo contact frame normal, hand->wall


class ContactProbe:
  """One CPU env with the root held fixed and the active arm driven by IK."""

  def __init__(self, hand: str = "left", wall_distance: float = 0.40, settle_steps: int = 25):
    self.hand = hand
    self.other = "right" if hand == "left" else "left"
    cfg = make_stable_contact_stand_env_cfg(active_hand=hand, wall_distance=wall_distance)
    cfg.scene.num_envs = 1
    cfg.events.pop("push_robot", None)
    self.env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    self.env.reset()
    self.robot = self.env.scene["robot"]
    self.site_cfg = SceneEntityCfg("robot", site_names=(G1_HANDS[hand].site,))
    self.site_cfg.resolve(self.env.scene)
    # The robot spawns with its feet above the floor; let it settle onto them
    # under zero action before freezing the root, so it stands on the ground.
    for _ in range(settle_steps):
      self.env.step(torch.zeros(1, self.env.action_manager.total_action_dim))
    self.root_state = torch.cat(
      [self.robot.data.root_link_pose_w, torch.zeros(1, 6)], dim=-1
    ).clone()

    self.mj_model = self.env.sim.mj_model
    self.mj_data = mujoco.MjData(self.mj_model)
    m = self.mj_model
    self.site_id = m.site(f"robot/{G1_HANDS[hand].site}").id
    joints = [m.joint(f"robot/{hand}_{j}_joint").id for j in ARM_JOINTS]
    self.dofs = [m.jnt_dofadr[j] for j in joints]
    self.qadr = [m.jnt_qposadr[j] for j in joints]
    names = list(self.robot.joint_names)
    self.action_idx = [names.index(f"{hand}_{j}_joint") for j in ARM_JOINTS]
    self.action_term = self.env.action_manager.get_term("joint_pos")
    self.hand_geom_id = m.geom(f"robot/{G1_HANDS[hand].geom}").id
    self.wall_geom_id = m.geom(f"{SURFACE}/{WALL_GEOM_NAME}").id
    self.hand_body_id = m.body(f"robot/{hand}_wrist_yaw_link").id
    self.raw_action = torch.zeros(1, len(names))
    self.push_w = torch.zeros(3)

  def wall_point_w(self, local: tuple[float, float, float]) -> np.ndarray:
    return mdp.surface_point_w(self.env, SURFACE, local)[0].numpy()

  def _sync_native(self) -> None:
    sim = self.env.sim.data
    self.mj_data.qpos[:] = sim.qpos[0].cpu().numpy()
    self.mj_data.qvel[:] = sim.qvel[0].cpu().numpy()
    self.mj_data.ctrl[:] = sim.ctrl[0].cpu().numpy()
    self.mj_data.mocap_pos[:] = sim.mocap_pos[0].cpu().numpy()
    self.mj_data.mocap_quat[:] = sim.mocap_quat[0].cpu().numpy()
    self.mj_data.qacc_warmstart[:] = sim.qacc_warmstart[0].cpu().numpy()
    self.mj_data.xfrc_applied[:] = sim.xfrc_applied[0].cpu().numpy()

  def set_palm_target(self, target_w: np.ndarray) -> None:
    """Damped least-squares IK on the active arm; sets joint-position actions."""
    self._sync_native()
    m, d = self.mj_model, self.mj_data
    for _ in range(200):
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      err = target_w - d.site_xpos[self.site_id]
      if np.linalg.norm(err) < 1e-4:
        break
      jacp = np.zeros((3, m.nv))
      mujoco.mj_jacSite(m, d, jacp, None, self.site_id)
      jac = jacp[:, self.dofs]
      dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(3), err)
      for adr, (lo, hi), v in zip(self.qadr, m.jnt_range[[m.dof_jntid[i] for i in self.dofs]], dq):
        d.qpos[adr] = np.clip(d.qpos[adr] + 0.5 * v, lo, hi)
    default = self.robot.data.default_joint_pos[0]
    scale = self.action_term.scale[0]
    for i, adr in zip(self.action_idx, self.qadr):
      self.raw_action[0, i] = (float(d.qpos[adr]) - default[i]) / scale[i]

  def push_into_wall(self, newtons: float) -> None:
    """Apply a known external force on the hand body along the inward wall normal."""
    outward = mdp.surface_normal_w(self.env, SURFACE, WALL_OUTWARD_NORMAL_LOCAL)[0]
    self.push_w = -outward * newtons

  def step(self, n: int = 1) -> None:
    for _ in range(n):
      self.robot.write_root_state_to_sim(self.root_state)
      self.env.sim.data.xfrc_applied[0, self.hand_body_id, :3] = self.push_w
      self.env.step(self.raw_action)

  def measure(self) -> Measurement:
    env, sensor = self.env, hand_sensor_name(self.hand)
    surface = {"surface_name": SURFACE, "normal_local": WALL_OUTWARD_NORMAL_LOCAL}
    force = mdp.hand_contact_force(env, sensor, **surface)[0]
    mu = float(max(self.mj_model.geom_friction[self.hand_geom_id, 0],
                   self.mj_model.geom_friction[self.wall_geom_id, 0]))
    ratio = mdp.friction_cone_ratio(force[0:1], force[1:2], mu)[0]
    self._sync_native()
    mujoco.mj_forward(self.mj_model, self.mj_data)
    native_normal = None
    for c in self.mj_data.contact[: self.mj_data.ncon]:
      if {c.geom1, c.geom2} == {self.hand_geom_id, self.wall_geom_id}:
        sign = 1.0 if c.geom1 == self.hand_geom_id else -1.0
        native_normal = sign * c.frame[:3].copy()
        break
    sid = self.site_cfg.site_ids[0]
    return Measurement(
      palm_w=self.robot.data.site_pos_w[0, sid].numpy().copy(),
      palm_vel_w=self.robot.data.site_lin_vel_w[0, sid].numpy().copy(),
      contact=bool(mdp.hand_contact_state(env, sensor)[0, 0]),
      normal_force=float(force[0]),
      tangential_force=float(force[1]),
      friction_cone_ratio=float(ratio),
      slip_speed=float(mdp.hand_slip_speed(env, sensor, **surface, asset_cfg=self.site_cfg)[0, 0]),
      contact_time=float(mdp.hand_contact_time(env, sensor)[0, 0]),
      other_hand_contact=bool(mdp.hand_contact_state(env, hand_sensor_name(self.other))[0, 0]),
      unintended_contacts=int(mdp.unintended_surface_contact(env, UNINTENDED_SENSOR)[0]),
      native_contact_normal=native_normal,
    )

  def close(self) -> None:
    self.env.close()


def run_phases(probe: ContactProbe, steps: int = 50) -> dict[str, list[Measurement]]:
  """Scripted phases. Targets are in the wall frame: (depth, along-wall, height),
  depth < 0 is off the wall. "Pressing harder" adds a known external push on the
  hand, because under joint-position control the contact force for a deeper
  joint target depends on the approach path (friction)."""
  fwd = G1_HANDS[probe.hand].forward_sign

  def run(name: str, target_local, push: float = 0.0, n: int = steps) -> None:
    probe.set_palm_target(probe.wall_point_w(target_local))
    probe.push_into_wall(push)
    out[name] = []
    for _ in range(n):
      probe.step()
      out[name].append(probe.measure())

  out: dict[str, list[Measurement]] = {}
  press = (LIGHT_PRESS_DEPTH, fwd * 0.15, -0.15)
  run("free", (-0.20, fwd * 0.15, -0.15))
  run("touch", (-HAND_CAPSULE_RADIUS, fwd * 0.15, -0.15))
  run("press", press)
  run("press+10N", press, push=10.0)
  run("press+20N", press, push=20.0)
  # Slide forward along the wall while pressing (commanded tangential motion).
  probe.push_into_wall(0.0)
  out["slide"] = []
  for forward in np.linspace(0.15, 0.35, steps):
    probe.set_palm_target(probe.wall_point_w((LIGHT_PRESS_DEPTH, fwd * forward, -0.15)))
    probe.step()
    out["slide"].append(probe.measure())
  return out


def summarize(results: dict[str, list[Measurement]], tail: int = 10) -> None:
  print(f"{'phase':<13}{'contact':>8}{'F_n[N]':>9}{'|F_t|[N]':>10}{'cone':>7}"
        f"{'slip[m/s]':>10}{'t_c[s]':>8}{'other':>7}{'unint':>7}  palm_w / native normal")
  for name, ms in results.items():
    window = ms[-tail:]
    mean = lambda f: float(np.mean([f(m) for m in window]))  # noqa: E731
    last = window[-1]
    normal = None if last.native_contact_normal is None else last.native_contact_normal.round(3)
    print(f"{name:<13}{mean(lambda m: m.contact):>8.2f}{mean(lambda m: m.normal_force):>9.2f}"
          f"{mean(lambda m: m.tangential_force):>10.2f}{mean(lambda m: m.friction_cone_ratio):>7.2f}"
          f"{mean(lambda m: m.slip_speed):>10.3f}{last.contact_time:>8.2f}"
          f"{mean(lambda m: m.other_hand_contact):>7.2f}{mean(lambda m: m.unintended_contacts):>7.2f}"
          f"  {last.palm_w.round(3)} / {normal}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--hand", choices=list(G1_HANDS), default="left")
  parser.add_argument("--steps", type=int, default=50)
  args = parser.parse_args()
  probe = ContactProbe(args.hand)
  print(f"wall outward normal (world): "
        f"{mdp.surface_normal_w(probe.env, SURFACE, WALL_OUTWARD_NORMAL_LOCAL)[0].numpy().round(3)}")
  summarize(run_phases(probe, args.steps))
  probe.close()


if __name__ == "__main__":
  main()

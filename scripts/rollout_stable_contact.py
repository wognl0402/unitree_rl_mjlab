"""Headless zero/random-policy rollout of Unitree-G1-StableContactMove.

Wiring check only, not a performance measure: prints contact, motion and
reward-component statistics so task plumbing can be verified before training.
Raw reward functions are evaluated directly (the reward manager replaces NaN
with 0, which would hide a broken term).

Run: python scripts/rollout_stable_contact.py --agent random --steps 300
"""

import argparse
import math
from collections import Counter

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
import src.tasks.stable_contact.mdp as mdp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from src.tasks.stable_contact.stable_contact_move_env_cfg import StableContactMoveCfg

TASK = "Unitree-G1-StableContactMove"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--agent", choices=("zero", "random"), default="zero")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--steps", type=int, default=300)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--displacement-range", type=float, default=None,
                      help="Override the curriculum (default: final stage) to exercise motion.")
  parser.add_argument("--log-every", type=int, default=25)
  args = parser.parse_args()

  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.curriculum = {}
  final_range = StableContactMoveCfg().displacement_stages[-1][1]
  cfg.commands["twist"].displacement_range = (
    final_range if args.displacement_range is None else args.displacement_range
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env.reset()
  torch.manual_seed(args.seed)
  n_act = env.action_manager.total_action_dim
  robot = env.scene["robot"]
  rm, mm, tm = env.reward_manager, env.metrics_manager, env.termination_manager
  term_counts: Counter[str] = Counter()
  episode_lengths: list[int] = []
  sums = torch.zeros(len(rm.active_terms))
  metric_sums = torch.zeros(len(mm.active_terms))

  print(f"task={TASK} agent={args.agent} envs={args.num_envs} steps={args.steps} "
        f"displacement_range={cfg.commands['twist'].displacement_range} m")
  header = ("step", "tgt_err", "contact", "F_n", "v_t", "cmd_v", "base_v", "tilt_deg", "reward")
  print(" ".join(f"{h:>8}" for h in header))
  for step in range(1, args.steps + 1):
    action = (torch.zeros(args.num_envs, n_act) if args.agent == "zero"
              else 2 * torch.rand(args.num_envs, n_act) - 1)
    lengths_before = env.episode_length_buf.clone()
    _, reward, terminated, truncated, _ = env.step(action.to(env.device))
    # Raw (pre-sanitization) reward terms must be finite.
    for name in rm.active_terms:
      cfg_t = rm.get_term_cfg(name)
      raw = cfg_t.func(env, **cfg_t.params)
      if not torch.isfinite(raw).all():
        raise RuntimeError(f"reward term {name} produced non-finite values at step {step}")
    sums += rm._step_reward.mean(dim=0).cpu()
    metric_sums += mm._step_values.mean(dim=0).cpu()
    done = (terminated | truncated).nonzero().flatten()
    for i in done.tolist():
      episode_lengths.append(int(lengths_before[i]) + 1)
      for name in tm.active_terms:
        if bool(tm.get_term(name)[i]):
          term_counts[name] += 1
    if step % args.log_every == 0:
      values = dict(zip(mm.active_terms, mm._step_values.mean(dim=0).tolist()))
      g = robot.data.projected_gravity_b
      tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1))).mean().item()
      row = (step, values["hand_target_error"], values["hand_contact_ratio"],
             values["hand_normal_force"], values["hand_tangential_speed"],
             values["command_speed"], values["base_speed"], tilt, reward.mean().item())
      print(f"{row[0]:>8d} " + " ".join(f"{v:>8.3f}" for v in row[1:]))

  print("\nmean per-step reward components (weight x value, before dt scaling):")
  for name, value in zip(rm.active_terms, (sums / args.steps).tolist()):
    weight = rm.get_term_cfg(name).weight
    print(f"  {name:<26} weight {weight:>8g}   mean {value:>9.4f}")
  print("\nmean metrics:")
  for name, value in zip(mm.active_terms, (metric_sums / args.steps).tolist()):
    print(f"  {name:<26} {value:>9.4f}")
  print(f"\nterminations: {dict(term_counts)}")
  if episode_lengths:
    mean_len = sum(episode_lengths) / len(episode_lengths)
    print(f"episodes ended: {len(episode_lengths)}, mean length {mean_len:.1f} steps "
          f"({mean_len * env.step_dt:.2f} s)")
  env.close()
  assert math.isfinite(float(sums.sum()))


if __name__ == "__main__":
  main()

"""Minimal policy <-> environment loop.

Builds a registered task's environment, builds (or loads) a policy, and loops:

  obs -> policy -> action -> env.step(action) -> obs, reward, terminated, ...

This is the same contract ``train.py`` relies on. Training adds rollout storage
and PPO updates around this loop; playing adds a viewer.

Run with:
  uv run python scripts/tutorial/policy_env_loop.py
  uv run python scripts/tutorial/policy_env_loop.py --num-steps 500 --num-envs 8
  uv run python scripts/tutorial/policy_env_loop.py \
    --checkpoint logs/rsl_rl/g1_velocity/<run>/model_<N>.pt
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import tyro
from tensordict import TensorDict

import mjlab
import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


@dataclass(frozen=True)
class LoopConfig:
  task: str = "Mjlab-Velocity-Flat-Unitree-G1"
  num_envs: int = 4
  num_steps: int = 200
  checkpoint: str | None = None
  """Optional ``model_<N>.pt``. Without it the policy has random weights."""
  device: str | None = None


def main(cfg: LoopConfig) -> None:
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # (ii) The environment: robot + scene + managers, built from the task config.
  env_cfg = load_env_cfg(cfg.task)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  # (i) The policy. The RSL-RL runner builds the actor network from the env's
  # observation/action sizes, so it needs the wrapped env once at construction.
  agent_cfg = load_rl_cfg(cfg.task)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  if cfg.checkpoint is not None:
    runner.load(cfg.checkpoint, load_cfg={"actor": True}, map_location=device)
  policy = runner.get_inference_policy(device=device)

  # The loop. obs is a dict of observation groups ("actor", "critic", ...),
  # each a tensor of shape [num_envs, group_dim].
  obs, _ = env.reset()
  for name, dim in env.observation_manager.group_obs_dim.items():
    print(f"obs['{name}']: ({env.num_envs}, {dim[0]})")
  print(f"action: {env.action_space.shape}")

  episode_return = torch.zeros(env.num_envs, device=device)
  with torch.inference_mode():
    for step in range(cfg.num_steps):
      # Policy: state -> action. The actor only reads the "actor" group.
      action = policy(TensorDict(obs, batch_size=[env.num_envs]))

      # Environment: action -> next state, reward, termination, aux info.
      obs, reward, terminated, truncated, extras = env.step(action)

      episode_return += reward
      done = terminated | truncated
      if done.any():
        ids = done.nonzero().squeeze(-1).tolist()
        returns = episode_return[done].tolist()
        print(f"step {step:4d}: envs {ids} reset, returns {returns}")
        episode_return[done] = 0.0  # Done envs were already auto-reset.
      if step % 50 == 0:
        print(
          f"step {step:4d}: mean reward {reward.mean().item():+.4f}, "
          f"extras keys {sorted(extras)}"
        )

  env.close()


if __name__ == "__main__":
  main(tyro.cli(LoopConfig, config=mjlab.TYRO_FLAGS))

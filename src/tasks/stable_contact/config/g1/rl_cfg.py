"""RL configuration for the G1 stable-contact move task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_stable_contact_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Same PPO settings as the G1 velocity task; separate experiment directory."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_stable_contact_move"
  return cfg

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from src.tasks.stable_contact.stable_contact_move_env_cfg import make_stable_contact_move_env_cfg

from .rl_cfg import unitree_g1_stable_contact_ppo_runner_cfg

# EXPERIMENTAL: fixed-point contact baseline (see stable_contact_move_env_cfg.py).
register_mjlab_task(
  task_id="Unitree-G1-StableContactMove",
  env_cfg=make_stable_contact_move_env_cfg(),
  play_env_cfg=make_stable_contact_move_env_cfg(play=True),
  rl_cfg=unitree_g1_stable_contact_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

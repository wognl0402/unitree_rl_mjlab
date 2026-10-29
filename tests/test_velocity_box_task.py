"""Tests for the G1 flat velocity task with a box observation."""

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_box_env_cfg


@pytest.mark.slow
def test_box_displacement_matches_world_distance() -> None:
  cfg = unitree_g1_flat_box_env_cfg()
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  env.step(torch.zeros(env.action_space.shape, device=env.device))

  obs = env.observation_manager.compute_group("actor")
  assert isinstance(obs, torch.Tensor)
  term_names = env.observation_manager.active_terms["actor"]
  assert term_names[-1] == "box_displacement"
  displacement = obs[:, -3:]

  robot = env.scene["robot"]
  torso_id = robot.find_bodies("torso_link")[0][0]
  torso_pos_w = robot.data.body_link_pos_w[:, torso_id]
  box_pos_w = env.scene["box"].data.root_link_pos_w
  # A rotation preserves length, so the body-frame vector has the world norm.
  torch.testing.assert_close(
    displacement.norm(dim=-1), (box_pos_w - torso_pos_w).norm(dim=-1)
  )
  env.close()

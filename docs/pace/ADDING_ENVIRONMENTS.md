# Find MDP functions and add an environment

This checkout includes the framework code used by the tasks. For example,
`mdp.last_action` comes from
[`envs/mdp/observations.py`](../../src/mjlab/envs/mdp/observations.py), while
velocity tracking rewards live in
[`tasks/velocity/mdp/rewards.py`](../../src/mjlab/tasks/velocity/mdp/rewards.py).
An MDP package can re-export functions from another package: follow the import
when a definition is not in the first file you open.

Complete the [README setup](../../README.md) and run `source env.sh` in this
checkout before using the commands below.

## Start with an existing task

For a new robot-learning task, define:

- A scene: robot model, terrain, sensors, and physics settings.
- Observations and actions: what the policy sees and controls.
- Commands, rewards, resets, and terminations: the behavior you want to learn.
- A training configuration: policy network and PPO settings.
- A unique registration ID, plus training and play configurations.

The simplest complete example is upstream's
[`cartpole` task](../../src/mjlab/tasks/cartpole) and its
[tutorial](../source/tutorials/cartpole.rst). The
[create-new-task notebook](../../notebooks/create_new_task.ipynb) provides another
walkthrough. For G1, start from
[`velocity/config/g1`](../../src/mjlab/tasks/velocity/config/g1).

## Example: register a slower G1 walking variant

The following is an exercise, not an additional task already registered in this
branch. Create a package directory `src/mjlab/tasks/g1_slow_walk/` and put this
in its `__init__.py`:

```python
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def slow_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.ranges.lin_vel_x = (0.0, 0.5)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  command.heading_command = False
  command.ranges.heading = None
  command.rel_world_envs = 0.0
  command.rel_forward_envs = 0.0
  cfg.curriculum.pop("command_vel", None)
  return cfg


runner_cfg = unitree_g1_ppo_runner_cfg()
runner_cfg.experiment_name = "g1_slow_walk"

register_mjlab_task(
  task_id="Mjlab-Velocity-Slow-Unitree-G1",
  env_cfg=slow_walk_env_cfg(),
  play_env_cfg=slow_walk_env_cfg(play=True),
  rl_cfg=runner_cfg,
  runner_cls=VelocityOnPolicyRunner,
)
```

The `twist` command samples requested forward, lateral, and turning velocities.
Disable both heading control and its heading range together. Disable world-frame
and forward-only sampling so those modes cannot reinterpret the requested range.
Remove the velocity curriculum so it cannot expand the range during training.
The inherited rewards still teach velocity tracking and balance.

`mjlab.tasks` recursively imports task packages, so the new `__init__.py`
registers the ID. First check discovery and configuration without simulation:

```bash
uv run list-envs --keyword Slow
uv run train Mjlab-Velocity-Slow-Unitree-G1 --help
```

Then, **inside a GPU allocation**, check zero actions and a small training run:

```bash
uv run play Mjlab-Velocity-Slow-Unitree-G1 \
  --agent zero --num-envs 1 --viewer viser
uv run train Mjlab-Velocity-Slow-Unitree-G1 \
  --env.scene.num-envs 32 --agent.max-iterations 2 \
  --agent.logger tensorboard --agent.run-name slow_walk_smoke
```

For a custom reward, add a Python function to your task's `mdp` package and
reference it in a `RewardTermCfg`. Read the existing
[`reward definitions`](../../src/mjlab/tasks/velocity/mdp/rewards.py) for tensor
shapes and entity selection. Evaluate the learned behavior in the viewer before
changing reward weights or increasing training time.

For tasks maintained outside the framework repository, mjlab also discovers
packages through the `mjlab.tasks` entry-point group; see
[`_import_registered_packages`](../../src/mjlab/__init__.py).

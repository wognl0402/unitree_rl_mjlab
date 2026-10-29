# RL basics and mjlab source map

This checkout contains the framework, robot tasks, and the MDP functions they use.
Work from one repository throughout the tutorial.

Follow the [main README](../../README.md) for the complete PACE setup and job
submission sequence. This page explains the ideas behind those commands.
Then use the [environment guide](ADDING_ENVIRONMENTS.md) to change a task.
The [new-environment tutorial](NEW_ENVIRONMENT_TUTORIAL.md) walks from the
policy/env loop through the managers to a registered task that spawns an
object and observes it.
The [G1 execution walkthrough and crawling plan](G1_EXECUTION_FLOW_AND_CRAWLING.md)
follows one example from task import through PPO and maps the required changes
for floor-based movement.

## What is reinforcement learning?

An **agent** learns by interacting with an **environment**. On each control step,
the policy reads observations and chooses actions. The simulator applies those
actions and advances physics. The environment returns new observations, a reward,
and a signal indicating whether the episode ended.

For a G1 walking task:

| RL concept | mjlab example |
| --- | --- |
| Environment | G1 robot, floor, physics, resets, and task rules |
| Observation | Joint positions and velocities, gravity direction, desired velocity |
| Action | Joint position targets, scaled before reaching the actuators |
| Reward | Track the requested velocity while avoiding unstable or wasteful movement |
| Episode | A simulation rollout ending at a failure or time limit |
| Policy | A neural network mapping observations to actions |
| Learning algorithm | PPO, implemented by the RSL-RL dependency |

**MDP** means *Markov decision process*. In mjlab, files named `mdp` contain the
functions that define observations, rewards, reset events, commands, and
termination conditions. These functions describe the task. PPO updates the policy
using data collected from that task; the reward function itself is not learned.

Training runs many simulated copies of the environment in parallel. PPO gathers
a batch of rollouts, estimates which actions led to better outcomes, and updates
the policy. An **iteration** contains many simulation steps and policy updates.
Two iterations are enough for a software smoke check, not a walking policy.

## Where the pieces live

| Piece | Source |
| --- | --- |
| Environment loop | [`manager_based_rl_env.py`](../../src/mjlab/envs/manager_based_rl_env.py) |
| Generic MDP functions | [`envs/mdp`](../../src/mjlab/envs/mdp) |
| Velocity task MDP functions | [`tasks/velocity/mdp`](../../src/mjlab/tasks/velocity/mdp) |
| G1 environment configuration | [`env_cfgs.py`](../../src/mjlab/tasks/velocity/config/g1/env_cfgs.py) |
| G1 PPO settings | [`rl_cfg.py`](../../src/mjlab/tasks/velocity/config/g1/rl_cfg.py) |
| Training entry point | [`train.py`](../../src/mjlab/scripts/train.py) |
| Upstream architecture guide | [`architecture_overview.rst`](../source/architecture_overview.rst) |

The [upstream docs](https://mujocolab.github.io/mjlab/) explain the general API.
This checkout also contains that documentation under `docs/source/` and the
[cartpole tutorial](../source/tutorials/cartpole.rst).

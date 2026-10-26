# Learn RL with mjlab on PACE-ICE

This checkout contains the framework, robot tasks, and the MDP functions they use.
Work from one repository throughout the tutorial.

1. [Set up Conda and PACE-ICE](SETUP_CONDA_ICE.md).
2. [Submit and monitor a GPU job](JOB_SUBMISSION.md).
3. [Find MDP functions and add an environment](ADDING_ENVIRONMENTS.md).

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

## Validation status

Checked on 2026-10-26 against the imported main revision:

- The locked source install, formatting, linting, and both type checkers passed.
- Thirteen upstream task/configuration tests passed on CPU. The documented
  environment exercise registered its training and play configurations.
- Mocked batch execution checked activation, arguments, GPU visibility, resume
  forwarding, and error propagation. PACE accepted the resource request with
  `sbatch --test-only` and no explicit scheduling-class argument.

These checks did not run GPU training or a fresh PACE Conda installation.
Complete the two-iteration batch smoke job before starting a longer run.

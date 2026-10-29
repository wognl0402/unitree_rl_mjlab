# Tutorial: from the RL loop to a new mjlab environment

This tutorial builds up in four steps:

1. How training is wired: a **runner** (the RL algorithm) and an **env** (robot
   and scene), and the contract between them.
2. A minimal working script that builds a policy and an environment and loops
   between them.
3. Inside the environment: `ManagerBasedRlEnv`, its managers, and what `step()`
   and `reset()` do.
4. Tasks and configs: where observations and rewards are declared, how to write
   a new observation function, and how to register a new task. The worked
   example spawns a box and observes it from the robot's torso.

Everything below runs in this checkout. The new task is already registered as
`Mjlab-Velocity-Flat-Box-Unitree-G1` so you can read it, run it, and copy it.

| File | What it is |
| --- | --- |
| [`scripts/tutorial/policy_env_loop.py`](../../scripts/tutorial/policy_env_loop.py) | Section 2: the minimal policy/env loop |
| [`tasks/velocity/mdp/observations.py`](../../src/mjlab/tasks/velocity/mdp/observations.py) | Section 4: `body_to_object_displacement` |
| [`tasks/velocity/config/g1/env_cfgs.py`](../../src/mjlab/tasks/velocity/config/g1/env_cfgs.py) | Section 4: `get_box_spec`, `unitree_g1_flat_box_env_cfg` |
| [`tasks/velocity/config/g1/__init__.py`](../../src/mjlab/tasks/velocity/config/g1/__init__.py) | Section 4: task registration |
| [`tests/test_velocity_box_task.py`](../../tests/test_velocity_box_task.py) | Check that the new observation is correct |

---

## 1. How RL training is wired

Open [`src/mjlab/scripts/train.py`](../../src/mjlab/scripts/train.py). Strip away
logging, W&B, video, and multi-GPU handling, and `run_train` does four things:

```python
env = ManagerBasedRlEnv(cfg=cfg.env, device=device)  # (ii) the env
env = RslRlVecEnvWrapper(env, clip_actions=...)  # adapt to RSL-RL
runner = runner_cls(env, agent_cfg, str(log_dir), device)  # (i) the runner
runner.learn(num_learning_iterations=cfg.agent.max_iterations)
```

Two objects, two configs:

| Piece | Class | Configured by | Contains |
| --- | --- | --- | --- |
| (i) Runner | `MjlabOnPolicyRunner` (subclass of RSL-RL's `OnPolicyRunner`) | `RslRlOnPolicyRunnerCfg`, e.g. [`g1/rl_cfg.py`](../../src/mjlab/tasks/velocity/config/g1/rl_cfg.py) | Actor and critic networks, PPO, rollout storage, logging, checkpoints |
| (ii) Env | `ManagerBasedRlEnv` | `ManagerBasedRlEnvCfg`, e.g. [`g1/env_cfgs.py`](../../src/mjlab/tasks/velocity/config/g1/env_cfgs.py) | Robot, scene, physics, observations, actions, rewards, resets, terminations |

`train.py` never mentions G1, walking, or PPO hyperparameters. It looks both
configs up by **task ID** (`load_env_cfg(task_id)`, `load_rl_cfg(task_id)`). That
is why the same script trains every task: a new scene is a new env config, and a
new algorithm is a new runner class or config. The script does not change.

### The contract

The env and the policy only meet through tensors. Every tensor has a leading
`num_envs` dimension because thousands of simulations run in parallel on the GPU.

```
              ┌──────────── action [N, A] ────────────┐
              │                                       ▼
        ┌──────────┐                           ┌─────────────┐
        │  policy  │                           │     env     │
        └──────────┘                           └─────────────┘
              ▲                                       │
              └───── obs {"actor": [N, O], ...} ──────┤
                                                      ├── reward      [N]
                                                      ├── terminated  [N]  (failure)
                                                      ├── truncated   [N]  (time limit)
                                                      └── extras      dict (logging)
```

- **Env:** `env.step(action) -> obs, reward, terminated, truncated, extras`.
- **Policy:** `policy(obs) -> action`. It does nothing else: no physics, no reward.

What the learner uses:

- `obs`, `action`, `reward`, and the done flags (`terminated | truncated`) are
  the RL data stored in the rollout buffer.
- `truncated` also matters to PPO: for an episode cut off by the time limit,
  RSL-RL bootstraps the value of the final state instead of treating it as a
  failure. The wrapper passes it on as `extras["time_outs"]`.
- The rest of `extras` (`extras["log"]`: per-term reward sums, termination
  counts, curriculum state) is for logging and debugging. It is what you see in
  TensorBoard or W&B, and it does not enter the PPO loss.

`obs` is a dict of **observation groups**. The velocity tasks define `"actor"`
(what the policy sees; noisy) and `"critic"` (what the value function sees;
noise-free plus privileged terms like foot contact). The actor network only
reads the `"actor"` group.

### What `runner.learn()` adds

RSL-RL's `OnPolicyRunner.learn` (in `.venv/.../rsl_rl/runners/on_policy_runner.py`)
is this loop:

```python
obs = env.get_observations()
for it in range(num_iterations):
    with torch.inference_mode():
        for _ in range(num_steps_per_env):        # collect a rollout
            actions = alg.act(obs)                # sample from the policy
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)   # store
        alg.compute_returns(obs)                  # advantages (GAE)
    alg.update()                                  # PPO gradient steps
    save checkpoint every save_interval
```

Take away storage, `compute_returns`, and `update`, and you are left with the
plain loop in the next section.

---

## 2. A minimal policy/env loop

[`scripts/tutorial/policy_env_loop.py`](../../scripts/tutorial/policy_env_loop.py)
loads a task, builds the policy, and loops. Its core:

```python
env_cfg = load_env_cfg(cfg.task)
env_cfg.scene.num_envs = cfg.num_envs
env = ManagerBasedRlEnv(cfg=env_cfg, device=device)  # spawn the env

agent_cfg = load_rl_cfg(cfg.task)
runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = runner_cls(wrapped, asdict(agent_cfg), device=device)  # sizes the nets
if cfg.checkpoint is not None:
  runner.load(cfg.checkpoint, load_cfg={"actor": True}, map_location=device)
policy = runner.get_inference_policy(device=device)  # load a policy

obs, _ = env.reset()
for step in range(cfg.num_steps):
  action = policy(TensorDict(obs, batch_size=[env.num_envs]))  # state -> action
  obs, reward, terminated, truncated, extras = env.step(action)
```

Run it (with a GPU; it falls back to CPU otherwise):

```bash
# Random-weight policy on the stock flat G1 task.
uv run python scripts/tutorial/policy_env_loop.py

# A trained checkpoint (must come from the same task: network sizes must match).
uv run python scripts/tutorial/policy_env_loop.py \
  --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint logs/rsl_rl/g1_velocity/<run>/model_<N>.pt
```

Output from a random policy on 4 envs (RTX 4090):

```
obs['actor']: (4, 99)
obs['critic']: (4, 111)
action: (4, 29)
step    0: mean reward +0.0889, extras keys ['log']
step   50: mean reward +0.0263, extras keys ['log']
step   74: envs [0] reset, returns [2.567286491394043]
step   77: envs [2] reset, returns [1.4908239841461182]
...
```

Things to notice:

- 29 actions: one joint-position target per G1 actuator.
- 99 actor observations: the concatenation of every term in the `"actor"` group
  (Section 3 shows the table the env prints at startup).
- Random weights make the robot fall after about 1.5 s. The `fell_over`
  termination fires, and the env **auto-resets** those envs inside `step()`.
  The `obs` you get back for them is already the first observation of the next
  episode. You never call `reset()` yourself in the loop.

---

## 3. Inside the environment: `ManagerBasedRlEnv`

[`src/mjlab/envs/manager_based_rl_env.py`](../../src/mjlab/envs/manager_based_rl_env.py)
does very little task-specific work itself. It owns a `Scene` (the entities and
sensors), a `Simulation` (MuJoCo Warp, all envs batched), and a set of
**managers**. Each manager reads one section of the config (a dict of named
*terms*) and calls each term's function.

| Manager | Config field | Term config | Called in |
| --- | --- | --- | --- |
| `ActionManager` | `actions` | `ActionTermCfg` (e.g. `JointPositionActionCfg`) | `step`: turn the policy output into actuator targets |
| `ObservationManager` | `observations` | `ObservationGroupCfg` → `ObservationTermCfg` | `step`, `reset`: build the obs dict |
| `RewardManager` | `rewards` | `RewardTermCfg` | `step`: weighted sum of terms |
| `TerminationManager` | `terminations` | `TerminationTermCfg` | `step`: which envs are done |
| `EventManager` | `events` | `EventTermCfg` (mode `startup`, `reset`, `interval`) | randomization, resets, pushes |
| `CommandManager` | `commands` | e.g. `UniformVelocityCommandCfg` | goals the policy must track |
| `CurriculumManager` | `curriculum` | `CurriculumTermCfg` | adjusts difficulty on reset |
| `MetricsManager` | `metrics` | `MetricsTermCfg` | logging only |

`load_managers()` builds them in dependency order (events, commands, actions,
observations, terminations, rewards, ...). Because the env only calls managers,
**adding behavior means adding a term to the config, not editing the env
class**.

### `step(action)`

```python
def step(self, action):
  self.action_manager.process_action(action)          # scale, offset, store

  for _ in range(self.cfg.decimation):                 # physics substeps
    self.action_manager.apply_action()                 # write actuator targets
    self.scene.write_data_to_sim()
    self.sim.step()                                    # MuJoCo step, all envs
    self.scene.update(dt=self.physics_dt)              # refresh sensors/buffers

  self.episode_length_buf += 1

  self.reset_buf = self.termination_manager.compute()  # which envs are done?
  self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

  self.event_manager.apply(mode="interval", ...)       # e.g. random pushes

  reset_env_ids = self.reset_buf.nonzero().squeeze(-1)
  if self.cfg.auto_reset and len(reset_env_ids) > 0:
    self._reset_idx(reset_env_ids)                     # reset only those envs

  self.sim.forward()                                   # refresh kinematics
  self.command_manager.compute(dt=self.step_dt)        # advance/resample goals
  self.sim.sense()
  self.obs_buf = self.observation_manager.compute(update_history=True)

  return self.obs_buf, self.reward_buf, terminated, time_outs, self.extras
```

The order is: apply action → simulate `decimation` physics steps → judge
(terminations, rewards) → reset finished envs → observe. For G1,
`timestep=0.005` and `decimation=4`, so one `step()` is 20 ms of simulated time
(50 Hz control) and four physics steps.

### `reset()` and `_reset_idx(env_ids)`

`_reset_idx` resets only the selected envs. It runs the curriculum, resets
sim and scene state to defaults, then fires every `mode="reset"` event (this is
where the robot is placed at a random pose, and where the box is placed in
Section 4). Then each manager clears its per-episode state and reports episode
sums into `extras["log"]`. `reset()` calls `_reset_idx` for all envs and returns
the first observation.

---

## 4. Tasks: configs, MDP functions, and a new environment

### Where a task is defined

A **task** is a registered pair of configs under an ID. For the flat G1 task,
[`g1/__init__.py`](../../src/mjlab/tasks/velocity/config/g1/__init__.py) does:

```python
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
```

The env config is built in layers. Each layer calls the previous one and edits
the result:

```
make_velocity_env_cfg()            velocity_env_cfg.py   robot-agnostic: obs, rewards,
        │                                                events, terminations, commands
        ▼
unitree_g1_rough_env_cfg()         g1/env_cfgs.py        G1 robot, foot sensors,
        │                                                body/site names, reward stds
        ▼
unitree_g1_flat_env_cfg()          g1/env_cfgs.py        plane terrain, no height scan
        │
        ▼
unitree_g1_flat_box_env_cfg()      g1/env_cfgs.py        + box, + box observation  (new)
```

To see which observations, rewards, and terminations a task uses, start at
[`velocity_env_cfg.py`](../../src/mjlab/tasks/velocity/velocity_env_cfg.py) and
then read the G1 overrides. Or run the env: every manager prints a table of its
active terms at startup (Section 4 shows the one for the new task).

### MDP functions: what a term actually calls

Each term config points at a plain Python function with `func=` and passes
extra arguments with `params=`. The manager calls `func(env, **params)` every
step. For an observation, the function returns a tensor shaped `[num_envs, X]`.

`foot_contact` in
[`velocity/mdp/observations.py`](../../src/mjlab/tasks/velocity/mdp/observations.py):

```python
def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()  # [num_envs, num_feet]
```

and its term in `velocity_env_cfg.py`:

```python
"foot_contact": ObservationTermCfg(
  func=mdp.foot_contact,
  params={"sensor_name": "feet_ground_contact"},
),
```

Reward terms (`RewardTermCfg`) return `[num_envs]` and carry a `weight`;
termination terms return a `[num_envs]` bool tensor. Generic functions live in
[`envs/mdp/`](../../src/mjlab/envs/mdp); task-specific ones in the task's own
`mdp/` package. `tasks/velocity/mdp/__init__.py` re-exports `envs.mdp`, so
`mdp.<name>` covers both.

### Reading scene state inside an MDP function

`env.scene["<name>"]` returns an entity (`Entity`) or a sensor. Entity state is
in `entity.data` (`EntityData`, [`entity/data.py`](../../src/mjlab/entity/data.py)).
All quantities are in the world frame (`_w`) unless suffixed `_b`, and batched
over envs:

| Property | Shape | Meaning |
| --- | --- | --- |
| `root_link_pos_w`, `root_link_quat_w` | `[N, 3]`, `[N, 4]` | Root body position, quaternion (w, x, y, z) |
| `root_link_lin_vel_b`, `root_link_ang_vel_b` | `[N, 3]` | Root velocity in the root frame |
| `body_link_pos_w`, `body_link_quat_w` | `[N, B, 3]`, `[N, B, 4]` | Every body of the entity |
| `body_link_lin_vel_w`, `body_com_pos_w`, ... | `[N, B, 3]` | Body velocities, center of mass |
| `site_pos_w`, `site_quat_w` | `[N, S, 3]`, `[N, S, 4]` | Sites (e.g. `left_foot`) |
| `joint_pos`, `joint_vel` | `[N, J]` | Joint state |
| `projected_gravity_b`, `heading_w` | `[N, 3]`, `[N]` | Orientation helpers |
| `default_root_state`, `default_joint_pos` | | Reset targets |

To select specific bodies, joints, sites, or geoms, pass a `SceneEntityCfg` in
`params`. The manager resolves its names (regex allowed) into indices once at
startup, so the function uses `asset_cfg.body_ids` / `joint_ids` / `site_ids`
without looking up names every step. Frame conversions are in
[`utils/lab_api/math.py`](../../src/mjlab/utils/lab_api/math.py)
(`quat_apply`, `quat_apply_inverse`, `yaw_quat`, ...).

G1 body names come from the robot model in
[`asset_zoo/robots`](../../src/mjlab/asset_zoo/robots). To print them:

```bash
uv run python -c "
from mjlab.asset_zoo.robots import get_g1_robot_cfg
print(get_g1_robot_cfg().build().body_names)"
```

Further examples to copy: `ee_to_object_distance` in
[`manipulation/mdp/observations.py`](../../src/mjlab/tasks/manipulation/mdp/observations.py)
(object relative to a site), and
[`velocity/mdp/rewards.py`](../../src/mjlab/tasks/velocity/mdp/rewards.py) for
bodies, sites, sensors, and commands.

### Worked example: spawn a box and observe it from the torso

Goal: a new task that is the flat G1 velocity task plus a box placed near the
robot each episode, and an observation of the box position relative to the
robot's `torso_link`, expressed in the torso frame.

Three pieces: an **entity** in the scene, a **reset event** to place it, and
an **observation function** plus its term.

#### Step 1: the observation function

Added to
[`velocity/mdp/observations.py`](../../src/mjlab/tasks/velocity/mdp/observations.py):

```python
def body_to_object_displacement(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Object root position relative to one robot body, in that body's frame.

  Returns:
    Tensor of shape [B, 3].
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  body_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids].squeeze(1)
  body_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids].squeeze(1)
  displacement_w = obj.data.root_link_pos_w - body_pos_w
  return quat_apply_inverse(body_quat_w, displacement_w)
```

- `asset_cfg.body_ids` selects one body, so `[N, 1, 3]` is squeezed to `[N, 3]`.
- The world-frame difference is rotated into the torso frame with the inverse
  torso rotation. A world-frame vector would change whenever the robot turned,
  even with the box in the same place relative to the robot. The body-frame
  vector is what the policy can act on.
- Because `mdp/__init__.py` star-imports this module, it is reachable as
  `mdp.body_to_object_displacement`.

#### Step 2: the object and the new config

Added to [`g1/env_cfgs.py`](../../src/mjlab/tasks/velocity/config/g1/env_cfgs.py):

```python
BOX_HALF_SIZE = 0.15


def get_box_spec() -> mujoco.MjSpec:
  """A free-floating box the robot can see (and bump into)."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(
    name="box_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(BOX_HALF_SIZE,) * 3,
    mass=2.0,
    rgba=(0.9, 0.5, 0.1, 1.0),
  )
  return spec


def unitree_g1_flat_box_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat G1 velocity task plus a box and a torso-to-box observation."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  # 1. Spawn the object. Its default position is relative to each env origin.
  cfg.scene.entities["box"] = EntityCfg(
    spec_fn=get_box_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(1.5, 0.0, BOX_HALF_SIZE)),
  )

  # 2. Re-place the box at every episode reset.
  cfg.events["reset_box"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"x": (-0.5, 0.5), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("box"),
    },
  )

  # 3. Observe the box from the torso. Both groups get it; the actor input
  # grows by 3, so checkpoints from the plain flat task are not compatible.
  box_obs = ObservationTermCfg(
    func=mdp.body_to_object_displacement,
    params={
      "object_name": "box",
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.observations["actor"].terms["box_displacement"] = box_obs
  cfg.observations["critic"].terms["box_displacement"] = box_obs

  return cfg
```

- **Entity.** An entity is anything built from an `MjSpec`: a robot from an
  XML file, or a body made in code like this one. The `freejoint` makes the box
  a floating body with physics (the robot can push it). The scene copies every
  entity into each of the `num_envs` worlds. The key in `scene.entities`
  (`"box"`) is the name used by `env.scene["box"]`.
- **Reset event.** `reset_root_state_uniform` sets the pose to
  `default pose + env origin + uniform sample`. Without it, the box would start
  at its default position every episode and keep wherever it was pushed only
  until the next reset. The robot's own `reset_base` event also randomizes its
  position and yaw, so the observed displacement varies widely.
- **Observation term.** The dict key (`"box_displacement"`) is the term name in
  logs and tables. The new term is appended to the end of each group.

Because the base task is reused, nothing else changes: the same velocity
command, rewards, and terminations apply. The box is an obstacle the policy can
now see. Making the robot *do* something with it (walk to it, push it) is the
next step: add a reward term using the same displacement, e.g. a negative
weight on its norm.

#### Step 3: register the task

Added to [`g1/__init__.py`](../../src/mjlab/tasks/velocity/config/g1/__init__.py):

```python
box_rl_cfg = unitree_g1_ppo_runner_cfg()
box_rl_cfg.experiment_name = "g1_velocity_box"

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Box-Unitree-G1",
  env_cfg=unitree_g1_flat_box_env_cfg(),
  play_env_cfg=unitree_g1_flat_box_env_cfg(play=True),
  rl_cfg=box_rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)
```

A separate `experiment_name` keeps its checkpoints in `logs/rsl_rl/g1_velocity_box/`,
apart from the plain flat task. `import mjlab.tasks` imports every task package,
so the ID is available to `train`, `play`, and `list-envs` with no further
wiring.

#### Step 4: check it

```bash
uv run list-envs --keyword Box

# The observation is correct (norm of body-frame vector = world distance).
uv run pytest tests/test_velocity_box_task.py

# The loop from Section 2 on the new task.
uv run python scripts/tutorial/policy_env_loop.py \
  --task Mjlab-Velocity-Flat-Box-Unitree-G1

# Look at it (zero actions; the box is orange).
uv run play Mjlab-Velocity-Flat-Box-Unitree-G1 --agent zero --num-envs 4

# Two-iteration training smoke run.
uv run train Mjlab-Velocity-Flat-Box-Unitree-G1 \
  --env.scene.num-envs 64 --agent.max-iterations 2 --agent.logger tensorboard
```

The observation manager's startup table shows the new term at the end of the
actor group, and the actor network input grows from 99 to 102:

```
|     7     | box_displacement                 |     (3,)    |
...
Actor Model: MLPModel(
    (0): Linear(in_features=102, out_features=512, bias=True)
```

The two-iteration run above completed on an RTX 4090 and wrote `model_0.pt`
and `model_1.pt`; `policy_env_loop.py --checkpoint .../model_1.pt` loaded the
result. That checks the plumbing only. It is not a trained policy.

### Checklist for your own environment

1. Pick the closest existing config and write a function that calls it and
   edits the result (`unitree_g1_flat_env_cfg` here).
2. New objects: an `MjSpec` builder plus `cfg.scene.entities["name"] = EntityCfg(...)`,
   and a `mode="reset"` event to place them.
3. New signals: a function `f(env, **params) -> Tensor[num_envs, X]` in the
   task's `mdp/` package, then an `ObservationTermCfg` / `RewardTermCfg` /
   `TerminationTermCfg` pointing at it.
4. Register a task ID with `register_mjlab_task`, with its own
   `experiment_name`.
5. Verify with `list-envs`, a zero-action `play`, the minimal loop, and a short
   `train` run before spending GPU hours.

# mjlab on PACE-ICE

Use this page from start to finish. This repository contains the complete
[mjlab framework](https://github.com/mujocolab/mjlab), including the MDP functions,
robot models, and Unitree G1/Go1 tasks. The example below trains G1 to follow
velocity commands on flat ground.

**One-time setup:** clone this branch, then `source env.sh`.
**Later sessions:** enter the same checkout and `source env.sh` again.
All environment setup and GPU checks are inside the root [`env.sh`](env.sh).

## 1. Log in to PACE — on your computer

Connect to the [Georgia Tech VPN](https://vpn.gatech.edu/global-protect/login.esp).
The course's access instructions require VPN even on the GT network. Replace
`YOUR_GT_USERNAME` with your Georgia Tech username:

```bash
ssh YOUR_GT_USERNAME@login-ice.pace.gatech.edu
```

For a web-based PACE session, use [ICE OnDemand](https://ondemand-ice.pace.gatech.edu/pun/sys/dashboard).
For a file browser, connect with SFTP to
`sftp://YOUR_GT_USERNAME@login-ice.pace.gatech.edu/`.

## 2. Clone and set up — on the PACE login node

Run this once:

```bash
mkdir -p ~/scratch
cd ~/scratch
git clone --branch codex/mjlab-upstream \
  https://github.com/wognl0402/unitree_rl_mjlab.git mjlab
cd mjlab
source env.sh
```

`env.sh` loads Conda, creates a Python 3.12 environment, installs this checkout
using `uv.lock`, puts caches in scratch, and activates the environment. The first
run downloads several GB and can take a few minutes. Wait for **`Ready:`**.
It does not start a simulation or request a GPU.

If `~/scratch/mjlab` already exists, inspect it before cloning. Use a different
directory name to keep an existing checkout. The helper detects its own location.

On every later login, use:

```bash
cd ~/scratch/mjlab
source env.sh
```

The default environment is `~/scratch/.conda/envs/mjlab-upstream-pace`. It is
separate from the earlier Unitree environment. There is no separate installer
to run and no second mjlab repository to clone. Do not install the old Unitree
`setup.py` package into this environment.

## 3. Submit a short GPU test — on the PACE login node

Stay in the repository root:

```bash
mkdir -p logs/slurm
JOB_ID=$(sbatch --parsable --time=00:15:00 \
  scripts/pace/train_g1.sbatch 32 2 smoke)
echo "Job ID: $JOB_ID"
```

The job script requests one GPU, four CPUs, and 32 GB RAM on `ice-gpu` with the
`cuda13` constraint. It uses your account's default scheduling settings. You do
not need an interactive allocation to submit it.

The three arguments are **parallel environments**, **PPO iterations**, and
**run name**. This example checks 32 environments for two iterations. It tests
the training pipeline; it does not teach useful walking.

## 4. Check the result — on the PACE login node

```bash
squeue -j "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed
tail -n 60 "logs/slurm/mjlab-g1-${JOB_ID}.out"
tail -n 60 "logs/slurm/mjlab-g1-${JOB_ID}.err"
```

- `PENDING`: waiting for resources. The log files may not exist yet.
- `RUNNING`: checking the GPU or training. First use can compile kernels.
- `COMPLETED` with `0:0`: the script finished successfully.
- `FAILED`, `TIMEOUT`, or `OUT_OF_MEMORY`: inspect both log files before retrying.

Use the printed numeric job ID after reconnecting; the `JOB_ID` shell variable
does not survive logout. To stop a job, run `scancel "$JOB_ID"`.

The job checks that mjlab is imported from this checkout, tests actual PyTorch
and Warp GPU allocations, and then runs PPO. Checkpoints are saved under:

```text
logs/rsl_rl/g1_velocity/<timestamp>_smoke_job<JOB_ID>/
```

## 5. Train longer — after the short test succeeds

```bash
JOB_ID=$(sbatch --parsable scripts/pace/train_g1.sbatch 1024 1000 g1_walk)
echo "Job ID: $JOB_ID"
```

This requests 1,024 environments and 1,000 iterations, with a one-hour job limit.
Training time and policy quality depend on the GPU and learning progress; this
is a teaching run, while the upstream G1 configuration defaults to 30,000
iterations. Increase the job time within your account limits when needed.

Scratch is temporary. Back up useful checkpoints and record the source commit.
Avoid changing a checkout or environment while its jobs are running.

## Optional: open the robot viewer

In your PACE terminal, request an interactive GPU and enter the compute node:

```bash
salloc --partition=ice-gpu \
  --constraint=cuda13 --gres=gpu:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=32G --time=01:00:00
srun --pty bash -l
cd ~/scratch/mjlab
source env.sh --check
hostname -s
uv run play Mjlab-Velocity-Flat-Unitree-G1 \
  --agent zero --num-envs 1 --viewer viser
```

Copy the compute hostname. Wait for Viser to print its port, usually `8080`.
This zero-action agent may fall; it is a viewer check.

In a **second terminal on your computer**, replace `COMPUTE_NODE` and your username:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 18080:COMPUTE_NODE:8080 \
  YOUR_GT_USERNAME@login-ice.pace.gatech.edu
```

Open <http://localhost:18080>. If Viser printed a different port, change the
final `8080` in the tunnel command. Keep both terminals open.

To view a trained policy, stop the zero-action viewer and replace its command
with this, using a real checkpoint path:

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint-file /FULL/PATH/TO/model_999.pt --num-envs 1 --viewer viser
```

When finished, stop the viewer with Ctrl+C, exit the compute shell, then exit
the allocation shell. Stop the tunnel with Ctrl+C in your computer's terminal.

## If setup fails

| Problem | Action |
| --- | --- |
| Conda is unavailable | Use a Bash login shell on PACE-ICE; the helper loads `anaconda3`. |
| Setup was interrupted or packages were changed | Run `source env.sh --install` on the login node. |
| Batch says setup is missing or outdated | Run `source env.sh` on the login node, then resubmit. |
| GPU check fails | Confirm you are inside a GPU compute allocation and read the error log. |
| Import or version errors | Reinstall with `source env.sh --install`; keep the checked-in lock. |
| Disk quota exceeded | Check free space in scratch; caches and the environment are stored there. |

`source env.sh --check` activates an existing installation and checks its GPU;
it never installs packages. The batch script uses this mode. Normal
`source env.sh` installs only when needed; `--install` forces a package reinstall
from the lock. These helpers do not edit your shell startup files or move home
directories.

## Understand and change the task

| What you want to find | File or guide |
| --- | --- |
| RL concepts and the training loop | [RL basics](docs/pace/README.md) |
| G1 observations, rewards, and commands | [Base velocity task](src/mjlab/tasks/velocity/velocity_env_cfg.py) and [G1 configuration](src/mjlab/tasks/velocity/config/g1/env_cfgs.py) |
| Generic MDP implementations | [`src/mjlab/envs/mdp/`](src/mjlab/envs/mdp) |
| Velocity-specific MDP implementations | [`src/mjlab/tasks/velocity/mdp/`](src/mjlab/tasks/velocity/mdp) |
| G1 policy network and PPO parameters | [`rl_cfg.py`](src/mjlab/tasks/velocity/config/g1/rl_cfg.py) |
| Add your own environment | [Environment guide](docs/pace/ADDING_ENVIRONMENTS.md) |
| Imported revision and branch history | [Source notes](docs/UPSTREAM.md) |

The framework is from [mujocolab/mjlab](https://github.com/mujocolab/mjlab), under
the [Apache 2.0 license](LICENSE). Its general documentation is included in
[`docs/source/`](docs/source), with citation details in [CITATION.cff](CITATION.cff).

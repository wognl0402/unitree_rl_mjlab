# PACE-ICE: Conda, mjlab, and the robot viewer

Use a terminal on your computer for SSH and a browser for Viser. Connect to the
Georgia Tech VPN before connecting to PACE when required by your access setup.
The original [course access page](https://chief-kryptops-30f.notion.site/PACE-setup-Conda-ICE-30e7204d85218000b10ffdb0cb1dcde3)
also introduces SSH, file transfer, and ICE OnDemand.

## 1. Connect and clone this branch

Replace `YOUR_GT_USERNAME` with your Georgia Tech username:

```bash
ssh YOUR_GT_USERNAME@login-ice.pace.gatech.edu
mkdir -p ~/scratch
cd ~/scratch
git clone --branch codex/mjlab-upstream \
  https://github.com/wognl0402/unitree_rl_mjlab.git mjlab
cd mjlab
```

If `~/scratch/mjlab` already exists, inspect that checkout before proceeding;
clone into a new directory if you want to keep it. All helpers derive the source
directory from their own checkout, so a different clone directory also works.

This is the **complete framework source**. Do not clone another mjlab repository
or install the old Unitree `setup.py` project alongside it.

## 2. Install once

On the login node, in the repository root:

```bash
bash scripts/pace/install.sh
source scripts/pace/env.sh
```

The installer loads PACE's `anaconda3` module if needed, creates a dedicated
Conda environment with Python 3.12 and uv, then installs this source checkout
using its `uv.lock` and CUDA 13.0 extra. It downloads packages but does not run a
simulation on the login node. First installation can take several minutes.

| Data | Default location |
| --- | --- |
| Source | `~/scratch/mjlab` |
| Conda environment | `~/scratch/.conda/envs/mjlab-upstream-pace` |
| Package and kernel caches | `~/scratch/.cache/` |
| Training checkpoints | `<checkout>/logs/rsl_rl/` |
| Slurm output | `<checkout>/logs/slurm/` |

Scratch storage is temporary. Copy important checkpoints and documents to
persistent storage. The helper sets cache paths without moving your existing
home directories or editing shell startup files.

For later sessions, run `source scripts/pace/env.sh` from this checkout.
The old `unitree-mjlab-pace` environment is separate; do not reuse it here.
Re-run the installer after intentional lockfile updates.

## 3. Enter a GPU compute node

In the same terminal:

```bash
salloc --partition=ice-gpu \
  --constraint=cuda13 --gres=gpu:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=32G --time=01:00:00
srun --pty bash -l
cd ~/scratch/mjlab
source scripts/pace/env.sh
hostname -s
uv run --frozen --no-sync python scripts/pace/check_environment.py
```

Wait for the allocation before entering the compute shell. The commands use
the account's default scheduling settings. `cuda13` selects compatible nodes;
Slurm chooses an eligible GPU. Copy the compute hostname for the tunnel below.
The check confirms the imported mjlab comes from this checkout and performs
small real allocations with both PyTorch and Warp.

## 4. Open the G1 viewer

On the compute node:

```bash
uv run --frozen --no-sync list-envs --keyword Unitree
uv run --frozen --no-sync play Mjlab-Velocity-Flat-Unitree-G1 \
  --agent zero --num-envs 1 --viewer viser
```

Wait for Viser to print its listening port, usually `8080`. A zero-action agent
checks simulation and viewing; it may fall and is not a learned walking policy.
First use may compile simulation kernels.

Open a **second terminal on your computer**, replace `COMPUTE_NODE` with the
hostname from step 3, and keep the tunnel running:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 18080:COMPUTE_NODE:8080 \
  YOUR_GT_USERNAME@login-ice.pace.gatech.edu
```

Visit <http://localhost:18080>. Press Play if the viewer is paused. If Viser
printed another port, change the final `8080` in the tunnel command to that port.

## 5. Check the training loop

Stop the viewer with Ctrl+C and run this on the compute node:

```bash
uv run --frozen --no-sync train Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 32 \
  --agent.max-iterations 2 \
  --agent.logger tensorboard \
  --agent.run-name interactive_smoke
```

Success means the run completes two PPO iterations and saves a checkpoint under
`logs/rsl_rl/g1_velocity/`. This checks the software pipeline; useful walking
requires longer training. The training command does not open the viewer.

Ctrl+C stops a running viewer. Exit the compute shell, then exit the allocation
shell to release the GPU. Stop the local tunnel with Ctrl+C. Use
[batch submission](JOB_SUBMISSION.md) for training that continues after logout.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No CUDA GPU | Enter the compute node with `srun`; source this checkout's environment. |
| Wrong mjlab source | Source `scripts/pace/env.sh` from the intended checkout and re-run its installer. |
| Import or version errors | Reinstall with the checked-in lock; avoid individually upgrading MuJoCo or Warp. |
| Quota or cache errors | Confirm the environment and cache paths point to scratch and check available space. |
| Viewer does not connect | Check the compute hostname, Viser's port, and both open terminals. |
| ECC or GPU hardware failure | Record the node and ask PACE support; request a different node. |
| First step takes a long time | Allow initial Warp kernel compilation to finish. |

The GPU package wheels provide CUDA runtimes; a separate CUDA module is not part
of this setup. An allocation and a compatible NVIDIA driver are still required.
Validation of older mjlab versions does not validate this main-branch snapshot;
run the smoke check above or the batch smoke job after installation.

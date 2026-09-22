# PACE-ICE: install once, open Viser, then train

This setup is for Linux x86_64, Python 3.11.16, and a CUDA GPU on Georgia Tech
PACE-ICE. It uses the versions in `requirements-pace.lock.txt`.

## 1. Terminal A: login and clone

Replace `YOUR_GT_USERNAME` with your account name. Use the GT VPN if your
connection requires it.

```bash
ssh YOUR_GT_USERNAME@login-ice.pace.gatech.edu
cd ~/scratch
git clone --branch codex/pace-quickstart \
  https://github.com/wognl0402/unitree_rl_mjlab.git
cd unitree_rl_mjlab
```

Keep the repo, environment, caches, and logs in scratch. Scratch is temporary;
back up useful checkpoints elsewhere. The scripts set cache paths explicitly
and do not move existing home directories or edit your `.bashrc`.

## 2. Enter a GPU compute node

```bash
salloc --partition=ice-gpu --qos=coe-grade \
  --constraint=cuda13 --gres=gpu:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G --time=01:00:00
srun --pty bash -l
hostname
```

`salloc` reserves resources; `srun` enters the compute node. Copy the hostname
for Terminal B. The constraint selects nodes carrying PACE's `cuda13` feature;
`gpu:1` allows any eligible GPU model. Your account must have access to the
requested partition and QoS.

## 3. Install and open the robot viewer

Run on the compute node:

```bash
cd ~/scratch/unitree_rl_mjlab
bash scripts/pace/install.sh
source scripts/pace/env.sh

python scripts/play.py Unitree-G1-Flat \
  --agent zero --num-envs 1 --viewer viser
```

The installer creates a separate environment at
`~/scratch/.conda/envs/unitree-mjlab-pace`, installs the pinned dependencies,
and checks real PyTorch/Warp CUDA allocations. The first install downloads
several GB. First use may also compile CUDA kernels.

Wait for `viser (listening *:8080)`. Keep Terminal A open. This is a zero-action
simulation check: no training, checkpoint, or W&B login is needed. It does not
demonstrate learned walking; falling or resetting is possible.

In future sessions, enter a GPU node, change to the repo, and run only:

```bash
source scripts/pace/env.sh
python scripts/play.py Unitree-G1-Flat --agent zero --num-envs 1 --viewer viser
```

## 4. Terminal B: connect your laptop browser

Open a **second terminal on your laptop**, not on PACE. Replace `COMPUTE_NODE`
with the hostname from Terminal A:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 18080:COMPUTE_NODE:8080 \
  YOUR_GT_USERNAME@login-ice.pace.gatech.edu
```

The command stays running silently. Open **http://localhost:18080** in your
laptop browser. Press Play in Viser if paused. If Viser printed 8081, replace
the final 8080 in the tunnel command with 8081. The local port stays 18080.

When finished, Ctrl+C stops the viewer and the tunnel. Exit the compute shell
and then the allocation shell to release the GPU.

## 5. A short training check

Stop Viser, then run in Terminal A:

```bash
python scripts/train.py Unitree-G1-Flat \
  --env.scene.num-envs 32 \
  --agent.max-iterations 2 \
  --agent.logger tensorboard \
  --agent.run-name pace_smoke
```

This ends after two PPO iterations and checks the training pipeline. It does
not produce a useful trained policy. Longer training is a separate step;
`train.py` does not automatically open Viser.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `warp` has no attribute `context` | Use Warp 1.12.0 with this mjlab 1.2.0 setup. Activate the pinned environment. |
| `mjtEnableBit` / `mjENBL_MULTICCD` import error | Keep MuJoCo 3.5.0 and mujoco-warp 3.5.0 together. Avoid individually upgrading either. |
| `pip check` reports `mjviser` needs MuJoCo >=3.6 | Use the clean environment. This mjlab release has its own Viser viewer and does not require `mjviser`. |
| CUDA is unavailable | Check that you entered the compute node with `srun`, then run `python scripts/pace/check_environment.py`. |
| Uncorrectable ECC error | Record the node and request another allocation with `--exclude=FAULTY_NODE`. Report it to PACE support. Reinstallation cannot fix a faulty GPU. |
| Viewer page refused | Keep both terminals open; check the compute hostname and the remote port printed by Viser. |
| Startup compiles kernels | Let first-use compilation finish. New package versions or GPU architectures can need a new cache. |
| Disk quota exceeded | Confirm `PACE_WORK`, cache variables, and the active environment point to scratch. |

The package wheels used in validation supplied their CUDA runtimes; no separate
CUDA module was loaded for the smoke test. A CUDA module is not a general fix
for driver, allocation, or GPU hardware problems. In particular,
`torch.cuda.is_available()` can be True on an ECC-faulted GPU; the tiny actual
allocation in `check_environment.py` catches that case.

## Validation and scope

Validated on PACE-ICE with an NVIDIA L40S on 2026-09-22:

- Python 3.11.16; mjlab 1.2.0; MuJoCo 3.5.0; mujoco-warp 3.5.0;
  Warp 1.12.0; PyTorch 2.14.0 (CUDA 13); SciPy 1.17.1; Viser 1.1.1.
- Actual PyTorch and Warp CUDA allocations passed.
- `Unitree-G1-Flat`, 32 environments, two training iterations: exit 0.
- Zero-agent Viser server: HTTP 200 and remained running until stopped.

The GPU test used an existing Python environment with Warp 1.12 in an isolated
directory. The lock records the required dependency graph, with version
constraints checked; unrelated packages were excluded. The complete
fresh-Conda installer has not yet been executed. Browser rendering through
each user's laptop tunnel still needs that user's check.

Original source revision: `1425b15f73bd4095f0df53709d7c389c3eb9e790`.
See [upstream tracking](UPSTREAM.md) before updating this checkout.

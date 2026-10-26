# Submit and monitor a PACE GPU training job

Complete [installation](SETUP_CONDA_ICE.md) first. Submit from a login node;
Slurm starts training on an allocated GPU node. You do not need an interactive
allocation to submit a batch job.

## 1. Prepare the checkout

```bash
cd ~/scratch/mjlab
source scripts/pace/env.sh
mkdir -p logs/slurm
```

Run `sbatch` from the repository root. Slurm opens output files before the script
runs, so `logs/slurm` must already exist. The script finds the environment helper
through `SLURM_SUBMIT_DIR`, not through Slurm's temporary copy of the script.

The checked-in [`train_g1.sbatch`](../../scripts/pace/train_g1.sbatch) requests
one GPU on `ice-gpu`, four CPUs, 32 GB RAM, and one hour. It selects `cuda13`
nodes and uses the account's default scheduling settings. It performs the GPU
preflight check, then trains `Mjlab-Velocity-Flat-Unitree-G1` using TensorBoard.

## 2. Check scheduling and submit a short test

Optional scheduler validation, which creates no running job:

```bash
sbatch --test-only scripts/pace/train_g1.sbatch
```

This checks the resource request; it does not check the environment, dependencies,
or GPU runtime. Use a small real job to check those:

```bash
SMOKE_JOB_ID=$(sbatch --parsable --time=00:15:00 \
  scripts/pace/train_g1.sbatch 32 2 batch_smoke)
echo "$SMOKE_JOB_ID"
squeue -j "$SMOKE_JOB_ID"
```

The script arguments are **number of environments**, **PPO iterations**, and
**run name**, in that order. A first run may spend several minutes compiling
kernels. Wait for `COMPLETED` and exit code `0:0` before scaling up.

## 3. Submit a longer run

```bash
TRAIN_JOB_ID=$(sbatch --parsable \
  scripts/pace/train_g1.sbatch 1024 1000 ice_g1)
echo "$TRAIN_JOB_ID"
```

With no arguments, these same values are the defaults. This is a starting
configuration, not a promise that training finishes within one hour or produces
a good policy. Adjust the time limit to your account and partition limits.
The upstream G1 configuration defaults to 30,000 iterations; 1,000 is a shorter
teaching run.

Do not change the package environment while jobs using it are running. Preserve
a copy of a run's commit ID and checkpoints before updating the checkout.

## 4. Read status and logs

```bash
squeue -u "$USER"
sacct -j "$TRAIN_JOB_ID" \
  --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
tail -f "logs/slurm/mjlab-g1-${TRAIN_JOB_ID}.out"
```

Ctrl+C stops `tail` without stopping training. Read the error file if a run fails:

```bash
tail -n 80 "logs/slurm/mjlab-g1-${TRAIN_JOB_ID}.err"
```

| State | Meaning |
| --- | --- |
| `PENDING` | Waiting for resources; inspect the reason shown by `squeue`. |
| `RUNNING` | The GPU check or training is executing. |
| `COMPLETED`, `0:0` | The script exited successfully. |
| `FAILED` | Inspect both output and error logs. |
| `TIMEOUT` | The requested time expired; resume from a saved checkpoint. |
| `OUT_OF_MEMORY` | Reduce environment count or request suitable resources. |

If you reconnect, paste the printed numeric job ID into commands in place of
`$TRAIN_JOB_ID`; shell variables do not persist across logins. To cancel a job:

```bash
scancel "$TRAIN_JOB_ID"
```

Checkpoints and TensorBoard events are under:

```text
logs/rsl_rl/g1_velocity/<timestamp>_ice_g1_job<JOB_ID>/
```

The script logs its source commit and package versions. Do not treat a scheduler
validation or a `PENDING` state as evidence of successful training.

## 5. Resume and replay

Use the exact saved directory and checkpoint names from your run:

```bash
sbatch scripts/pace/train_g1.sbatch 1024 1000 ice_g1_resume \
  --agent.resume True \
  --agent.load-run 'YOUR_RUN_DIRECTORY_NAME' \
  --agent.load-checkpoint 'model_999.pt'
```

Extra arguments are forwarded to mjlab. Resume restores the saved policy and
optimizer into a new run directory; the iteration argument controls the requested
training iterations for the resumed invocation.

To view a checkpoint, request a fresh interactive GPU allocation using the setup
guide, source the helper, and run on that compute node:

```bash
uv run --frozen --no-sync play Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint-file /FULL/PATH/TO/model_999.pt \
  --num-envs 1 --viewer viser
```

Use the setup guide's SSH tunnel to open Viser. Replay does not submit a new
training job.

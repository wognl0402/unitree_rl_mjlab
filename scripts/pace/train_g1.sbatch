#!/bin/bash -l
#SBATCH --job-name=mjlab-g1
#SBATCH --partition=ice-gpu
#SBATCH --constraint=cuda13
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Submit from the repository root after creating logs/slurm.
# Arguments: NUM_ENVS ITERATIONS RUN_NAME [additional train arguments...]
set -eo pipefail
: "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root.}"
: "${SLURM_JOB_ID:?Submit this script with sbatch.}"

num_envs="${1:-1024}"
iterations="${2:-1000}"
run_name="${3:-pace_g1}"
if (( $# >= 3 )); then shift 3; else set --; fi
if [[ ! "$num_envs" =~ ^[1-9][0-9]*$ || ! "$iterations" =~ ^[1-9][0-9]*$ ]]; then
  echo 'NUM_ENVS and ITERATIONS must be positive integers.' >&2
  exit 2
fi
if [[ ! "$run_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo 'Use letters, digits, underscores, dots, or hyphens in RUN_NAME.' >&2
  exit 2
fi

source "$SLURM_SUBMIT_DIR/env.sh" --check
set -u
cd "$MJLAB_REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
printf 'Job %s | Host %s | Envs %s | Iterations %s\n' \
  "$SLURM_JOB_ID" "$(hostname -s)" "$num_envs" "$iterations"
git rev-parse HEAD
exec srun uv run train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs "$num_envs" \
  --agent.max-iterations "$iterations" \
  --agent.logger tensorboard \
  --agent.run-name "${run_name}_job${SLURM_JOB_ID}" \
  --agent.seed 7 \
  --agent.save-interval 50 \
  "$@"

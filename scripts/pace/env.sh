#!/usr/bin/env bash
# Source this from an interactive compute-node shell.
export PACE_WORK="${PACE_WORK:-$HOME/scratch}"
export PACE_ENV="$PACE_WORK/.conda/envs/unitree-mjlab-pace"
export XDG_CACHE_HOME="$PACE_WORK/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export CONDA_PKGS_DIRS="$PACE_WORK/.conda/pkgs"
export CONDA_ENVS_DIRS="$PACE_WORK/.conda/envs"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export WANDB_DIR="$PACE_WORK/wandb"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_MODE=offline
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" \
  "$WARP_CACHE_PATH" "$WANDB_DIR" "$WANDB_CACHE_DIR" || return 1

if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3 || return 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh" || return 1
if [[ "${1:-}" != --no-activate ]]; then
  if [[ ! -x "$PACE_ENV/bin/python" ]]; then
    echo 'First run: bash scripts/pace/install.sh' >&2
    return 1
  fi
  conda activate "$PACE_ENV" || return 1
fi

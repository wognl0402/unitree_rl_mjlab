#!/usr/bin/env bash
# Source this file from the checkout you intend to use.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source scripts/pace/env.sh [--no-activate]' >&2
  exit 2
fi

export MJLAB_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MJLAB_WORK="${MJLAB_WORK:-$HOME/scratch}"
export MJLAB_ENV="${MJLAB_ENV:-$MJLAB_WORK/.conda/envs/mjlab-upstream-pace}"
export CONDA_ENVS_PATH="$MJLAB_WORK/.conda/envs"
export CONDA_PKGS_DIRS="$MJLAB_WORK/.conda/pkgs"
export XDG_CACHE_HOME="$MJLAB_WORK/.cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PROJECT_ENVIRONMENT="$MJLAB_ENV"
export UV_PYTHON="$MJLAB_ENV/bin/python"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export WANDB_DIR="$MJLAB_WORK/wandb"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_MODE=offline
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$UV_CACHE_DIR" \
  "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WARP_CACHE_PATH" \
  "$MPLCONFIGDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" || return 1

if ! command -v conda >/dev/null 2>&1; then
  module load anaconda3 || return 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh" || return 1

case "${1:-}" in
  --no-activate) ;;
  ""|--activate)
    if [[ ! -x "$MJLAB_ENV/bin/python" ]]; then
      echo 'Run bash scripts/pace/install.sh first.' >&2
      return 1
    fi
    conda activate "$MJLAB_ENV" || return 1
    ;;
  *) echo 'Use: source scripts/pace/env.sh [--no-activate]' >&2; return 2 ;;
esac

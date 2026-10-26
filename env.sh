#!/usr/bin/env bash
# Source from a Bash shell. This file contains all PACE setup and GPU checks.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source env.sh [--install | --check]' >&2
  exit 2
fi

_mjlab_env_main() {
  local mode="${1:-}" digest expected state ready=0
  local -a sync_flags=()
  case "$mode" in
    ""|--install|--check) ;;
    --help)
      echo 'source env.sh           Install if needed, then activate.'
      echo 'source env.sh --install Reinstall locked packages, then activate.'
      echo 'source env.sh --check   Activate an existing setup and test its GPU.'
      return 0
      ;;
    *) echo 'Use: source env.sh [--install | --check]' >&2; return 1 ;;
  esac

  export MJLAB_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  if [[ ! -f "$MJLAB_REPO/uv.lock" || ! -d "$MJLAB_REPO/src/mjlab" ]]; then
    echo 'Keep env.sh at the root of the mjlab source checkout.' >&2
    return 1
  fi
  export MJLAB_WORK="${MJLAB_WORK:-$HOME/scratch}"
  export MJLAB_ENV="${MJLAB_ENV:-$MJLAB_WORK/.conda/envs/mjlab-upstream-pace}"
  export CONDA_ENVS_DIRS="$MJLAB_WORK/.conda/envs"
  export CONDA_PKGS_DIRS="$MJLAB_WORK/.conda/pkgs"
  export XDG_CACHE_HOME="$MJLAB_WORK/.cache"
  export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
  export UV_PROJECT_ENVIRONMENT="$MJLAB_ENV"
  export UV_PYTHON="$MJLAB_ENV/bin/python"
  # Commands such as `uv run train` use the environment prepared below.
  export UV_NO_SYNC=1
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

  mkdir -p "$CONDA_ENVS_DIRS" "$CONDA_PKGS_DIRS" "$UV_CACHE_DIR" \
    "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WARP_CACHE_PATH" \
    "$MPLCONFIGDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" || return 1

  if ! command -v conda >/dev/null 2>&1; then
    if ! type module >/dev/null 2>&1; then
      echo 'Conda is unavailable. Run this in a PACE-ICE Bash login shell.' >&2
      return 1
    fi
    module load anaconda3 || return 1
  fi
  # Conda treats these as aliases and rejects configurations containing both.
  unset CONDA_ENVS_PATH
  local conda_base
  conda_base="$(conda info --base)" || return 1
  source "$conda_base/etc/profile.d/conda.sh" || return 1

  digest="$(sha256sum "$MJLAB_REPO/pyproject.toml" "$MJLAB_REPO/uv.lock")" || return 1
  expected="$(printf '%s\n%s\n' "$MJLAB_REPO" "$digest")"
  state="$MJLAB_ENV/.mjlab-source-state"
  if [[ -x "$MJLAB_ENV/bin/python" && -x "$MJLAB_ENV/bin/uv" \
        && -f "$state" && "$(cat "$state")" == "$expected" ]]; then
    ready=1
  fi

  if [[ "$mode" == --check && "$ready" != 1 ]]; then
    echo 'Setup is missing or outdated. Run source env.sh on the login node first.' >&2
    return 1
  fi
  if [[ "$mode" == --install || "$ready" != 1 ]]; then
    # A failed repair must not leave an earlier successful setup marked ready.
    rm -f "$state" || return 1
    if [[ ! -x "$MJLAB_ENV/bin/python" ]]; then
      conda create --yes --prefix "$MJLAB_ENV" --override-channels \
        --channel conda-forge python=3.12 uv || return 1
    fi
    if [[ ! -x "$MJLAB_ENV/bin/uv" ]]; then
      conda install --yes --prefix "$MJLAB_ENV" --override-channels \
        --channel conda-forge uv || return 1
    fi
    if [[ "$mode" == --install ]]; then
      sync_flags+=(--reinstall)
    fi
    "$MJLAB_ENV/bin/uv" sync --project "$MJLAB_REPO" --locked --inexact \
      --no-dev --extra cu130 --python "$MJLAB_ENV/bin/python" \
      "${sync_flags[@]}" || return 1
    printf '%s\n' "$expected" > "$state" || return 1
  fi
  conda activate "$MJLAB_ENV" || return 1
  printf 'Ready: %s\nEnvironment: %s\n' "$MJLAB_REPO" "$MJLAB_ENV"

  if [[ "$mode" == --check ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
      echo 'Run --check on a GPU compute node inside a Slurm allocation.' >&2
      return 1
    fi
    "$MJLAB_ENV/bin/uv" run --project "$MJLAB_REPO" --no-sync \
      python - "$MJLAB_REPO" <<'PY' || return 1
import os
import sys
from importlib.metadata import version
from pathlib import Path

import torch
import warp as wp

import mjlab

source = Path(mjlab.__file__).resolve()
if source != Path(sys.argv[1]) / "src/mjlab/__init__.py":
  raise SystemExit(f"Wrong mjlab source: {source}. Run source env.sh --install.")
print(f"Source: {source}")
for package in ("mjlab", "torch", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib"):
  print(f"{package}: {version(package)}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
if not torch.cuda.is_available():
  raise SystemExit("PyTorch cannot see an allocated CUDA GPU.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
tensor = torch.zeros(16, device="cuda:0")
torch.cuda.synchronize()
wp.init()
array = wp.zeros(16, device="cuda:0")
wp.synchronize()
print(f"PASS: PyTorch {tensor.device}, Warp {array.device}")
PY
  fi
}

if _mjlab_env_main "$@"; then
  unset -f _mjlab_env_main
else
  unset -f _mjlab_env_main
  return 1
fi

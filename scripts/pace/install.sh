#!/usr/bin/env bash
set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh" --no-activate
if [[ ! -x "$MJLAB_ENV/bin/python" ]]; then
  conda create --yes --prefix "$MJLAB_ENV" --channel conda-forge python=3.12 uv
fi
source "$MJLAB_REPO/scripts/pace/env.sh" --activate
set -u

if ! command -v uv >/dev/null 2>&1; then
  conda install --yes --prefix "$MJLAB_ENV" --channel conda-forge uv
fi
cd "$MJLAB_REPO"
# Keep Conda's uv and seed packages while installing the locked source package.
uv sync --locked --inexact --no-dev --extra cu130 --python "$MJLAB_ENV/bin/python"
printf '\nInstalled mjlab from %s\n' "$MJLAB_REPO"
printf 'Enter a GPU compute node before running the environment check or training.\n'

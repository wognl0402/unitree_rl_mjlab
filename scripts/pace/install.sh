#!/usr/bin/env bash
set -euo pipefail
PACE_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PACE_REPO"
source scripts/pace/env.sh --no-activate
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == login* ]]; then
  echo 'Enter the GPU compute node first: salloc ..., then srun --pty bash -l' >&2
  exit 1
fi
if [[ ! -x "$PACE_ENV/bin/python" ]]; then
  conda create --yes --prefix "$PACE_ENV" python=3.11.16
fi
"$PACE_ENV/bin/python" -m pip install -r requirements-pace.lock.txt
"$PACE_ENV/bin/python" -m pip install --no-deps -e .
"$PACE_ENV/bin/python" -m pip check
"$PACE_ENV/bin/python" scripts/pace/check_environment.py
echo 'Ready. Run: source scripts/pace/env.sh'

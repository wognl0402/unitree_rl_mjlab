"""Check PACE shell setup without installing packages or allocating a GPU."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FAKE_CONDA = r"""#!/usr/bin/python3
import json, os, pathlib, shutil, sys
args = sys.argv[1:]
with open(os.environ["PACE_TEST_CALLS"], "a") as stream:
    stream.write(json.dumps({"program": "conda", "args": args}) + "\n")
if args == ["info", "--base"]:
    if os.environ.get("CONDA_ENVS_PATH") and os.environ.get("CONDA_ENVS_DIRS"):
        sys.exit(19)
    print(os.environ["PACE_TEST_CONDA_BASE"])
else:
    if os.environ.get("PACE_TEST_CONDA_FAIL"):
        sys.exit(13)
    prefix = pathlib.Path(args[args.index("--prefix") + 1]) / "bin"
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "python").write_text("#!/bin/bash\nexit 0\n")
    (prefix / "python").chmod(0o755)
    shutil.copy(os.environ["PACE_TEST_UV_TEMPLATE"], prefix / "uv")
"""

FAKE_UV = r"""#!/usr/bin/python3
import json, os, sys
args = sys.argv[1:]
entry = {"program": "uv", "args": args, "cwd": os.getcwd(),
         "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
         "threads": os.environ.get("OMP_NUM_THREADS"),
         "srun": os.environ.get("PACE_TEST_IN_SRUN"),
         "no_sync": os.environ.get("UV_NO_SYNC"),
         "source_env": os.environ.get("UV_PROJECT_ENVIRONMENT")}
if args[0] == "sync":
    code = os.environ.get("PACE_TEST_SYNC_EXIT", "0")
elif "train" in args:
    code = os.environ.get("PACE_TEST_TRAIN_EXIT", "0")
else:
    entry["stdin"] = sys.stdin.read()
    code = os.environ.get("PACE_TEST_CHECK_EXIT", "0")
with open(os.environ["PACE_TEST_CALLS"], "a") as stream:
    stream.write(json.dumps(entry) + "\n")
sys.exit(int(code))
"""


@pytest.fixture
def pace_setup(tmp_path: Path) -> tuple[Path, dict[str, str]]:
  checkout = tmp_path / "checkout with spaces"
  helpers = checkout / "scripts/pace"
  helpers.mkdir(parents=True)
  (checkout / "src/mjlab").mkdir(parents=True)
  shutil.copy(ROOT / "env.sh", checkout / "env.sh")
  shutil.copy(ROOT / "scripts/pace/train_g1.sbatch", helpers / "train_g1.sbatch")
  (checkout / "pyproject.toml").write_text("project for setup test\n")
  (checkout / "uv.lock").write_text("lock for setup test\n")
  bin_dir = tmp_path / "bin"
  bin_dir.mkdir()
  for name, text in {
    "conda": FAKE_CONDA,
    "uv-template": FAKE_UV,
    "git": "#!/bin/bash\necho test-source-commit\n",
    "srun": '#!/bin/bash\nexport PACE_TEST_IN_SRUN=1\nexec "$@"\n',
  }.items():
    path = bin_dir / name
    path.write_text(text)
    path.chmod(0o755)
  base = tmp_path / "conda-base"
  (base / "etc/profile.d").mkdir(parents=True)
  (base / "etc/profile.d/conda.sh").write_text(
    "conda() {\n"
    '  if [[ "$1" == activate ]]; then\n'
    '    export CONDA_PREFIX="$2"\n'
    '    export PATH="$2/bin:$PATH"\n'
    "  else\n"
    '    "$PACE_TEST_BIN/conda" "$@"\n'
    "  fi\n"
    "}\n"
  )
  work = tmp_path / "scratch with spaces"
  env = dict(
    os.environ,
    PATH=f"{bin_dir}:/usr/bin:/bin",
    PACE_TEST_BIN=str(bin_dir),
    PACE_TEST_REPO=str(checkout),
    PACE_TEST_CONDA_BASE=str(base),
    PACE_TEST_UV_TEMPLATE=str(bin_dir / "uv-template"),
    PACE_TEST_CALLS=str(tmp_path / "calls.jsonl"),
    MJLAB_WORK=str(work),
    MJLAB_ENV=str(work / ".conda/envs/mjlab-upstream-pace"),
    CUDA_VISIBLE_DEVICES="3",
  )
  env.pop("SLURM_JOB_ID", None)
  return checkout, env


def run_setup(
  env: dict[str, str], *args: str, module: bool = False
) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
  calls = Path(env["PACE_TEST_CALLS"])
  calls.unlink(missing_ok=True)
  shell = "set -u\n"
  if module:
    shell += (
      'module() { [[ "$*" == "load anaconda3" ]] || return 1; '
      'export PATH="$PACE_TEST_BIN:$PATH"; }\n'
    )
  shell += (
    'before="$-"\n'
    'if source "$PACE_TEST_REPO/env.sh" "$@"; then\n'
    '  [[ "$-" == "$before" ]] || exit 90\n'
    '  [[ "${CONDA_PREFIX:-}" == "$MJLAB_ENV" ]] || exit 91\n'
    "  ! declare -F _mjlab_env_main >/dev/null || exit 92\n"
    "else\n"
    '  exit "$?"\n'
    "fi\n"
  )
  result = subprocess.run(
    ["bash", "-c", shell, "pace-test", *args],
    env=env,
    cwd=Path(env["PACE_TEST_REPO"]).parent,
    capture_output=True,
    text=True,
  )
  rows = (
    [json.loads(line) for line in calls.read_text().splitlines()]
    if calls.exists()
    else []
  )
  return result, rows


def test_first_install_reuse_repair_and_lock_change(pace_setup):
  checkout, env = pace_setup
  result, calls = run_setup(env)
  assert result.returncode == 0, result.stderr
  assert "Ready:" in result.stdout
  assert any(row["args"][0] == "create" for row in calls)
  sync = [row for row in calls if row["program"] == "uv"]
  assert len(sync) == 1 and "--locked" in sync[0]["args"]
  assert "cu130" in sync[0]["args"] and "--inexact" in sync[0]["args"]
  assert (Path(env["MJLAB_ENV"]) / ".mjlab-source-state").exists()

  result, calls = run_setup(env)
  assert result.returncode == 0, result.stderr
  assert all(row["program"] != "uv" for row in calls)

  result, calls = run_setup(env, "--install")
  assert result.returncode == 0, result.stderr
  assert any("--reinstall" in row["args"] for row in calls)

  (checkout / "uv.lock").write_text("changed lock\n")
  result, calls = run_setup(env)
  assert result.returncode == 0, result.stderr
  assert any(row["args"][0] == "sync" for row in calls)


def test_loads_conda_module_when_needed(pace_setup):
  _, env = pace_setup
  result, _ = run_setup(dict(env, PATH="/usr/bin:/bin"), module=True)
  assert result.returncode == 0, result.stderr


def test_normalizes_inherited_conda_directory_aliases(pace_setup):
  _, env = pace_setup
  result, _ = run_setup(
    dict(env, CONDA_ENVS_PATH="/legacy/envs", CONDA_ENVS_DIRS="/module/envs")
  )
  assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("failure", ["PACE_TEST_CONDA_FAIL", "PACE_TEST_SYNC_EXIT"])
def test_failed_install_does_not_mark_environment_ready(pace_setup, failure):
  _, env = pace_setup
  result, _ = run_setup(dict(env, **{failure: "13"}))
  assert result.returncode != 0
  assert not (Path(env["MJLAB_ENV"]) / ".mjlab-source-state").exists()


def test_check_never_installs_and_requires_allocation(pace_setup):
  _, env = pace_setup
  result, calls = run_setup(env, "--check")
  assert result.returncode != 0 and "login node first" in result.stderr
  assert all(row["args"][0] not in {"create", "install", "sync"} for row in calls)
  result, _ = run_setup(env)
  assert result.returncode == 0, result.stderr
  result, calls = run_setup(env, "--check")
  assert result.returncode != 0 and "Slurm allocation" in result.stderr
  assert all(row["program"] != "uv" for row in calls)
  result, calls = run_setup(dict(env, SLURM_JOB_ID="123"), "--check")
  assert result.returncode == 0, result.stderr
  check = [row for row in calls if row["program"] == "uv"]
  assert len(check) == 1 and check[0]["args"][0] == "run"
  compile(check[0]["stdin"], "env.sh GPU check", "exec")


def test_failed_repair_invalidates_ready_marker(pace_setup):
  _, env = pace_setup
  result, _ = run_setup(env)
  assert result.returncode == 0, result.stderr
  result, _ = run_setup(dict(env, PACE_TEST_SYNC_EXIT="13"), "--install")
  assert result.returncode != 0
  assert not (Path(env["MJLAB_ENV"]) / ".mjlab-source-state").exists()
  result, calls = run_setup(dict(env, SLURM_JOB_ID="123"), "--check")
  assert result.returncode != 0 and "login node first" in result.stderr
  assert all(row["program"] != "uv" for row in calls)


@pytest.mark.parametrize(
  ("extra", "expected", "training"),
  [
    ({}, 0, True),
    ({"PACE_TEST_CHECK_EXIT": "13"}, 1, False),
    ({"PACE_TEST_TRAIN_EXIT": "17"}, 17, True),
  ],
)
def test_batch_checks_gpu_and_preserves_training_exit(
  pace_setup, extra, expected, training
):
  checkout, env = pace_setup
  result, _ = run_setup(env)
  assert result.returncode == 0, result.stderr
  calls_path = Path(env["PACE_TEST_CALLS"])
  calls_path.unlink()
  resume = ["--agent.resume", "True", "--agent.load-run", "previous_run"]
  result = subprocess.run(
    [
      "bash",
      str(checkout / "scripts/pace/train_g1.sbatch"),
      "32",
      "2",
      "smoke",
      *resume,
    ],
    env=dict(
      env,
      SLURM_SUBMIT_DIR=str(checkout),
      SLURM_JOB_ID="123",
      SLURM_CPUS_PER_TASK="4",
      **extra,
    ),
    cwd=checkout.parent,
    capture_output=True,
    text=True,
  )
  assert result.returncode == expected, result.stderr
  calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
  train = [row for row in calls if "train" in row["args"]]
  assert bool(train) == training
  if training:
    assert train[0]["args"][:3] == ["run", "train", "Mjlab-Velocity-Flat-Unitree-G1"]
    assert train[0]["args"][-len(resume) :] == resume
    assert "smoke_job123" in train[0]["args"]
    assert train[0]["cwd"] == str(checkout)
    assert train[0]["cuda"] == "3" and train[0]["threads"] == "4"
    assert train[0]["srun"] == "1" and train[0]["no_sync"] == "1"

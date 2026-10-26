"""Check the source checkout and actual GPU allocations inside a PACE job."""

import os
from importlib.metadata import version
from pathlib import Path

import torch
import warp as wp

import mjlab


def main() -> None:
  if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("Run this check inside a GPU allocation, not on a login node.")

  repo = Path(__file__).resolve().parents[2]
  source = Path(mjlab.__file__).resolve()
  if source != repo / "src" / "mjlab" / "__init__.py":
    raise SystemExit(f"Wrong mjlab source: {source}. Activate this checkout's env.")

  print(f"mjlab source: {source}", flush=True)
  for package in ("mjlab", "torch", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib"):
    print(f"{package}: {version(package)}", flush=True)
  print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

  if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see an allocated CUDA GPU.")
  print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
  tensor = torch.zeros(16, device="cuda:0")
  torch.cuda.synchronize()
  wp.init()
  array = wp.zeros(16, device="cuda:0")
  wp.synchronize()
  print(f"PASS: PyTorch {tensor.device}, Warp {array.device}", flush=True)


if __name__ == "__main__":
  main()

"""Small hardware/import check. No training and no checkpoint needed."""
import importlib.metadata as md
import os
import socket
import sys

expected = {
    "mjlab": "1.2.0", "mujoco": "3.5.0", "mujoco-warp": "3.5.0",
    "warp-lang": "1.12.0", "scipy": "1.17.1",
}
print(f"Node: {socket.gethostname()} | Slurm job: {os.getenv('SLURM_JOB_ID', 'none')}")
for name, want in expected.items():
    try:
        actual = md.version(name)
    except md.PackageNotFoundError:
        sys.exit(f"Missing {name}; run bash scripts/pace/install.sh.")
    print(f"{name}={actual}")
    if actual != want:
        sys.exit(f"Expected {name}={want}; activate the pinned PACE environment.")
try:
    import torch
    import mujoco
    import mujoco_warp
    import warp as wp

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable. Confirm you ran srun --pty bash -l after salloc.")
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch={torch.__version__}")
    x = torch.zeros(4, device="cuda:0")
    torch.cuda.synchronize()
    wp.init()
    y = wp.zeros(4, device="cuda:0")
    wp.synchronize()
except Exception as exc:
    if "ECC" in str(exc):
        sys.exit(f"GPU hardware error on {socket.gethostname()}: {exc}\n"
                 "Request another node; reinstalling packages will not fix this GPU.")
    raise
print("PASS: package versions, imports, PyTorch CUDA allocation, and Warp allocation.")

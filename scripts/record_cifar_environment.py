"""Record the exact DGX Python/GPU environment for a future CIFAR freeze.

Run in a short gpu_student SLURM allocation; never on the login node. This
does not launch training or alter the preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
from pathlib import Path

import numpy as np
import torch
import torchvision


def runtime_fingerprint() -> dict:
    """Return the runtime fields that a confirmation job must reproduce."""
    if not torch.cuda.is_available():
        raise ValueError("A CUDA device is required for the CIFAR runtime fingerprint.")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_compute_capability": [properties.major, properties.minor],
        "gpu_multiprocessor_count": properties.multi_processor_count,
        "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def environment_record() -> dict:
    if socket.gethostname().startswith("dgx-login"):
        raise ValueError("Environment capture must not run on the DGX login node.")
    if (not os.environ.get("SLURM_JOB_ID") or
            os.environ.get("SLURM_JOB_PARTITION") != "gpu_student" or
            not torch.cuda.is_available()):
        raise ValueError("Environment capture requires an active gpu_student allocation.")
    return {
        "schema_version": "cifar-dgx-environment-v2",
        "runtime": runtime_fingerprint(),
        "capture": {
            "hostname": socket.gethostname(),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_job_gres": os.environ.get("SLURM_JOB_GRES"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Environment manifest already exists; do not overwrite it.")
    try:
        record = environment_record()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(record, indent=2, sort_keys=True,
                              allow_nan=False) + "\n").encode()
        args.output.write_bytes(payload)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"CIFAR environment capture failed: {exc}\n")
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(payload).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

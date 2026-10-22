#!/usr/bin/env python3
"""Write a JSON environment and GPU audit without changing the environment."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = ["torch", "transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes", "numpy"]


def run(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:  # audit should record unavailable commands too
        return {"command": command, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None

    torch_info: dict[str, object] = {"installed": versions["torch"] is not None}
    if versions["torch"]:
        try:
            import torch

            torch_info.update(
                {
                    "version": torch.__version__,
                    "compiled_cuda": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                    "devices": [
                        {
                            "index": index,
                            "name": torch.cuda.get_device_name(index),
                            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                            "capability": list(torch.cuda.get_device_capability(index)),
                        }
                        for index in range(torch.cuda.device_count())
                    ],
                }
            )
        except Exception as exc:
            torch_info["import_error"] = repr(exc)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "packages": versions,
        "torch": torch_info,
        "nvidia_smi": run(["nvidia-smi"]),
        "nvidia_query": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]),
        "wsl_release": run(["bash", "-lc", "cat /etc/os-release; uname -a"]),
    }
    output = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

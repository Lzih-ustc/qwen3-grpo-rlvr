from __future__ import annotations

import subprocess
import threading
import time


def query_gpu() -> dict[str, float | None]:
    fields = "memory.used,memory.total,utilization.gpu,temperature.gpu"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        values = [float(part.strip()) for part in result.stdout.strip().splitlines()[0].split(",")]
        return dict(zip(["memory_used_mib", "memory_total_mib", "gpu_util_percent", "temperature_c"], values, strict=True))
    except Exception:
        return {"memory_used_mib": None, "memory_total_mib": None, "gpu_util_percent": None, "temperature_c": None}


class GpuMonitor:
    """Sample whole-device nvidia-smi data; torch reports process-local allocator stats."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = query_gpu()
            sample["timestamp_monotonic"] = time.monotonic()
            self.samples.append(sample)
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=6)

    def summary(self) -> dict[str, float | int | None]:
        used = [sample["memory_used_mib"] for sample in self.samples if sample["memory_used_mib"] is not None]
        util = [sample["gpu_util_percent"] for sample in self.samples if sample["gpu_util_percent"] is not None]
        return {
            "nvidia_smi_samples": len(self.samples),
            "whole_gpu_peak_memory_used_mib": max(used) if used else None,
            "whole_gpu_peak_utilization_percent": max(util) if util else None,
            "whole_gpu_start_memory_used_mib": used[0] if used else None,
            "whole_gpu_end_memory_used_mib": used[-1] if used else None,
        }

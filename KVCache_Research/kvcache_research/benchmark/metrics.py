from __future__ import annotations

from dataclasses import dataclass, asdict
import torch


@dataclass
class MemoryStats:
    avg_allocated_mb: float
    peak_allocated_mb: float


class MemoryTracker:
    def __init__(self, device: str) -> None:
        self.device = device
        self._samples: list[int] = []
        self._peak = 0

    def record(self) -> None:
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated(self.device)
        self._samples.append(allocated)
        self._peak = max(self._peak, allocated)

    def summary(self) -> MemoryStats:
        if not self._samples:
            return MemoryStats(avg_allocated_mb=0.0, peak_allocated_mb=0.0)
        avg = sum(self._samples) / len(self._samples)
        return MemoryStats(avg_allocated_mb=avg / (1024 * 1024), peak_allocated_mb=self._peak / (1024 * 1024))


@dataclass
class RunMetrics:
    ttft_s: float
    tpot_s: float
    tbt_s: float
    memory: MemoryStats

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["memory"] = asdict(self.memory)
        return payload

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable
import torch


@dataclass
class CacheEntry:
    past_key_values: tuple
    device: str
    num_bytes: int


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _kv_bytes(past_key_values: tuple) -> int:
    total = 0
    for key, value in past_key_values:
        total += _tensor_bytes(key)
        total += _tensor_bytes(value)
    return total


def _move_kv(past_key_values: tuple, device: str) -> tuple:
    return tuple((k.to(device), v.to(device)) for k, v in past_key_values)


class KVCacheStore:
    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._gpu_bytes = 0

    def has(self, request_id: str) -> bool:
        return request_id in self._entries

    def get(self, request_id: str, device: str) -> tuple | None:
        if request_id not in self._entries:
            return None
        entry = self._entries[request_id]
        if entry.device != device:
            entry = self._move_entry(request_id, entry, device)
        self._touch(request_id)
        return entry.past_key_values

    def set(self, request_id: str, past_key_values: tuple, device: str) -> None:
        num_bytes = _kv_bytes(past_key_values)
        if request_id in self._entries:
            self._remove_entry(request_id)
        entry = CacheEntry(past_key_values=past_key_values, device=device, num_bytes=num_bytes)
        self._entries[request_id] = entry
        self._touch(request_id)
        if device.startswith("cuda"):
            self._gpu_bytes += num_bytes

    def offload_lru(self, max_gpu_bytes: int, protect: Iterable[str] | None = None) -> None:
        if max_gpu_bytes <= 0:
            return
        protect_ids = set(protect or [])
        while self._gpu_bytes > max_gpu_bytes:
            victim_id = None
            for candidate_id in self._lru.keys():
                if candidate_id in protect_ids:
                    continue
                entry = self._entries[candidate_id]
                if entry.device.startswith("cuda"):
                    victim_id = candidate_id
                    break
            if victim_id is None:
                break
            self._move_entry(victim_id, self._entries[victim_id], "cpu")

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "gpu_bytes": self._gpu_bytes,
        }

    def _touch(self, request_id: str) -> None:
        if request_id in self._lru:
            self._lru.move_to_end(request_id)
        else:
            self._lru[request_id] = None

    def _remove_entry(self, request_id: str) -> None:
        entry = self._entries.pop(request_id)
        self._lru.pop(request_id, None)
        if entry.device.startswith("cuda"):
            self._gpu_bytes -= entry.num_bytes

    def _move_entry(self, request_id: str, entry: CacheEntry, device: str) -> CacheEntry:
        moved = CacheEntry(
            past_key_values=_move_kv(entry.past_key_values, device),
            device=device,
            num_bytes=entry.num_bytes,
        )
        self._entries[request_id] = moved
        if entry.device.startswith("cuda"):
            self._gpu_bytes -= entry.num_bytes
        if device.startswith("cuda"):
            self._gpu_bytes += entry.num_bytes
        return moved

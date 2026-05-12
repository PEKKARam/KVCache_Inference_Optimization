import time
from contextlib import contextmanager

import torch


@contextmanager
def timed(label, timings):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    timings[label] = end - start


def reset_cuda_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_cuda_peak_memory_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    return allocated, reserved


def sync_if_cuda(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

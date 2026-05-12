import time
from dataclasses import dataclass
from typing import List

import torch
from vllm import LLM, SamplingParams

from .metrics import reset_cuda_peak_memory, get_cuda_peak_memory_mb, sync_if_cuda


@dataclass
class VLLMBenchmarkResult:
    batch_size: int
    input_len: int
    output_len: int
    latency_s: float
    tokens_per_s: float
    peak_mem_alloc_mb: float
    peak_mem_reserved_mb: float


def run_vllm_benchmark(
    model_name,
    prompts: List[str],
    batch_size,
    input_len,
    output_len,
    dtype,
    tensor_parallel_size,
    gpu_memory_utilization,
    max_num_seqs,
    enforce_eager=False,
):
    batch_prompts = prompts[:batch_size]
    sampling = SamplingParams(max_tokens=output_len, temperature=0.0)

    llm = LLM(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
    )

    reset_cuda_peak_memory()
    sync_if_cuda("cuda")
    start = time.perf_counter()
    _ = llm.generate(batch_prompts, sampling)
    sync_if_cuda("cuda")
    end = time.perf_counter()

    latency = end - start
    total_tokens = batch_size * output_len
    tps = total_tokens / latency if latency > 0 else 0.0
    peak_alloc, peak_reserved = get_cuda_peak_memory_mb()

    return VLLMBenchmarkResult(
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        latency_s=latency,
        tokens_per_s=tps,
        peak_mem_alloc_mb=peak_alloc,
        peak_mem_reserved_mb=peak_reserved,
    )

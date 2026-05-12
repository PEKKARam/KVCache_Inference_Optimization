import time
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import reset_cuda_peak_memory, get_cuda_peak_memory_mb, sync_if_cuda


@dataclass
class HFBenchmarkResult:
    batch_size: int
    input_len: int
    output_len: int
    latency_s: float
    tokens_per_s: float
    peak_mem_alloc_mb: float
    peak_mem_reserved_mb: float


def run_hf_benchmark(
    model_name,
    prompts: List[str],
    batch_size,
    input_len,
    output_len,
    device,
    dtype,
    use_cache=True,
    attn_implementation="sdpa",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )

    batch_prompts = prompts[:batch_size]
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=input_len,
    ).to(device)

    reset_cuda_peak_memory()
    sync_if_cuda(device)
    start = time.perf_counter()
    _ = model.generate(
        **inputs,
        max_new_tokens=output_len,
        use_cache=use_cache,
        do_sample=False,
    )
    sync_if_cuda(device)
    end = time.perf_counter()

    latency = end - start
    total_tokens = batch_size * output_len
    tps = total_tokens / latency if latency > 0 else 0.0
    peak_alloc, peak_reserved = get_cuda_peak_memory_mb()

    return HFBenchmarkResult(
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        latency_s=latency,
        tokens_per_s=tps,
        peak_mem_alloc_mb=peak_alloc,
        peak_mem_reserved_mb=peak_reserved,
    )

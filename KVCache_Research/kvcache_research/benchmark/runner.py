from __future__ import annotations

import time
from typing import Iterable
import torch

from kvcache_research.benchmark.metrics import MemoryTracker, RunMetrics
from kvcache_research.cache.store import KVCacheStore


def _ensure_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def run_prompt(
    model,
    tokenizer,
    prompt: str,
    cache_store: KVCacheStore | None,
    cache_policy: str,
    max_gpu_bytes: int,
    request_id: str,
    device: str,
    max_new_tokens: int,
) -> RunMetrics:
    device = _ensure_device(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    past_key_values = None
    if cache_store is not None:
        past_key_values = cache_store.get(request_id, device)

    start_time = time.time()
    ttft = None
    total_token_time = 0.0
    memory = MemoryTracker(device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            step_start = time.time()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            if cache_store is not None:
                cache_store.set(request_id, past_key_values, device)
                if cache_policy == "lru_offload":
                    cache_store.offload_lru(max_gpu_bytes=max_gpu_bytes, protect=[request_id])

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = next_token
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

            if ttft is None:
                ttft = time.time() - start_time
            total_token_time += time.time() - step_start
            memory.record()

    tbt = time.time() - start_time
    tpot = total_token_time / max_new_tokens
    return RunMetrics(ttft_s=ttft or 0.0, tpot_s=tpot, tbt_s=tbt, memory=memory.summary())


def run_benchmark(
    model,
    tokenizer,
    prompts: Iterable[str],
    cache_policy: str,
    max_gpu_bytes: int,
    reuse_cache: bool,
    device: str,
    max_new_tokens: int,
) -> list[RunMetrics]:
    cache_store = KVCacheStore() if cache_policy != "none" else None
    metrics = []
    for idx, prompt in enumerate(prompts):
        request_id = f"request-{idx}" if not reuse_cache else "shared-request"
        metrics.append(
            run_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                cache_store=cache_store,
                cache_policy=cache_policy,
                max_gpu_bytes=max_gpu_bytes,
                request_id=request_id,
                device=device,
                max_new_tokens=max_new_tokens,
            )
        )
    return metrics

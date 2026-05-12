# KV Cache Inference Experiments

This folder contains runnable demos and benchmarks for:
- Transformer autoregressive decoding (token-by-token) with and without KV cache.
- KV cache management logic and simulation of PagedAttention and Continuous Batching.
- Qwen inference benchmarks using HuggingFace Transformers vs vLLM.

## Layout
- src/
  - kvcache_demo.py: minimal Transformer with KV cache support.
  - kvcache_manager.py: KV cache manager and paging simulation.
  - hf_benchmark.py: HuggingFace benchmark helpers.
  - vllm_benchmark.py: vLLM benchmark helpers.
  - metrics.py: timing and GPU memory metrics.
  - utils.py: misc utilities.
- scripts/
  - run_torch_kvcache_demo.py: token-by-token decoding demo.
  - simulate_kv_cache_manager.py: KV cache manager simulation.
  - run_hf_benchmark.py: HuggingFace Qwen benchmark.
  - run_vllm_benchmark.py: vLLM Qwen benchmark.
- configs/
  - benchmark.yaml: recommended benchmark settings.
  - prompts.txt: sample prompts.

## Quickstart
1. Create env with PyTorch, Transformers, vLLM, and PyYAML.
2. Adjust model path and GPU settings if needed.
3. Run the scripts:

```bash
python scripts/run_torch_kvcache_demo.py
python scripts/simulate_kv_cache_manager.py
python scripts/run_hf_benchmark.py --config configs/benchmark.yaml
python scripts/run_vllm_benchmark.py --config configs/benchmark.yaml
```

## What to Look For
- Token-by-token decoding is much faster with KV cache because attention only attends to new tokens and reuses previous keys/values.
- PagedAttention reduces memory fragmentation by storing KV in fixed-size blocks and mapping logical sequences to physical blocks.
- Continuous Batching improves throughput by merging variable-length requests into a single batch while keeping per-request KV state.

## Notes on PagedAttention and Continuous Batching
- PagedAttention uses a block allocator to map each sequence to a set of KV blocks. This makes it possible to evict or swap blocks and reduces wasted space from padding.
- Continuous Batching reuses a single execution loop, inserting new requests as others finish. The scheduler keeps KV cache consistent by tracking per-request positions.

## Expected Metrics
- Latency: time per request and per generated token.
- Throughput: total tokens per second.
- GPU memory usage: peak allocated and reserved memory.

## Tips
- For accurate GPU memory metrics, run with `torch.cuda.reset_peak_memory_stats()` before each run.
- Use fixed seeds and warmup iterations for stable latency numbers.

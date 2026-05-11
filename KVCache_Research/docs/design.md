# Design Notes

## Core Idea
The project provides a clean PyTorch baseline for KV cache management with
correct `past_key_values` semantics. The focus is on cache policy design and
memory observability rather than custom kernels.

## Cache Semantics
- Each request stores the full `past_key_values` tuple from the last decode step.
- Before the next forward pass, the cached KV is moved to the model device.
- This avoids partial or step-sliced caches that can break correctness.

## Policies
- `none`: baseline (no offload, no cache reuse across requests)
- `lru_offload`: offload least-recently-used request caches to CPU when the
  GPU cache budget is exceeded

## Metrics
- TTFT: time to first token
- TPOT: average time per token
- TBT: total time per request
- GPU memory: average and peak `memory_allocated`

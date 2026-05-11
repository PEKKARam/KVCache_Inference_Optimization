# Resume Points (Draft)

- Built a pure PyTorch KV cache research scaffold with correct `past_key_values`
  semantics and pluggable cache policies.
- Implemented LRU-based KV cache offloading with GPU memory budgeting and
  per-request cache tracking.
- Delivered reproducible latency/memory benchmarks (TTFT, TPOT, peak GPU mem)
  with configurable prompts and model settings.

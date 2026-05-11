# KVCache Research (Pure PyTorch)

This is a focused research project for KV cache management strategies in PyTorch.
It emphasizes correct cache semantics, clear baselines, and reproducible benchmarks
rather than vLLM-level kernel optimizations.

## Goals
- Correct `past_key_values` semantics for decoder-only models
- Pluggable cache policies (baseline vs. LRU offload)
- Lightweight, reproducible benchmarks
- Resume-friendly project structure and documentation

## Layout
- kvcache_research/   Core library (cache, model loader, benchmark)
- configs/            JSON configs and prompt samples
- scripts/            Entrypoints for benchmarks
- docs/               Design notes and resume points
- KVM/                Legacy code (kept for reference)
- Baseline/           Legacy code (kept for reference)

## Quickstart
1) Install dependencies
   - `pip install -r requirements.txt`

2) Run baseline benchmark
   - `python scripts/run_benchmark.py --config configs/baseline.json`

3) Run LRU offload benchmark
   - `python scripts/run_benchmark.py --config configs/offload_lru.json`

## Notes
- This project is a research scaffold. It does not attempt to replicate vLLM kernels.
- GPU memory numbers are captured via `torch.cuda.memory_allocated`.

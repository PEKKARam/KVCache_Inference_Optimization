# Inference Service Benchmark Report

## Environment
- GPU:
- Driver:
- CUDA:
- OS:
- Framework: vLLM

## Experiment Design
- Model: Qwen2.5-1.5B-Instruct
- Variables:
  - batch size / max_num_seqs
  - gpu_memory_utilization
  - quantization (FP16 / INT8 / INT4)
- Metrics:
  - QPS
  - latency (p50/p90/p99)
  - GPU memory usage

## Results
- Table: key metrics by configuration
- Charts: latency and throughput vs configuration

## Pitfalls and Fixes
- WSL2 networking and proxy
- Model download latency
- GPU memory tuning

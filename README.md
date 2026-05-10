# KVCache Inference Optimization (AI Infra)

This project benchmarks local LLM inference services with vLLM and compares
latency, throughput, and GPU memory usage across configurations and quantization
methods.

## Layout
- infra/        vLLM service configs and start script
- bench/        wrk and locust load tests
- experiments/  experiment configs, prompts, results
- tools/        GPU monitoring scripts
- src/          benchmark runner
- reports/      report template

## Quickstart (WSL2)
1) Start vLLM server
   - cd infra/vllm
   - MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" ./serve.sh

2) Run a baseline client benchmark
   - python src/runner.py --config experiments/configs/baseline.yaml

3) Run wrk
   - cd bench/wrk
   - URL="http://127.0.0.1:8000" ./run_wrk.sh

4) Run Locust (optional)
   - cd bench/locust
   - HOST="http://127.0.0.1:8000" ./run_locust.sh

5) Log GPU usage
   - python tools/gpu_monitor.py --out experiments/results/gpu.csv --duration 60

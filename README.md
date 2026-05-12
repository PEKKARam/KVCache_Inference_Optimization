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

## 中文简介
本项目用于对本地 vLLM 推理服务做基准测试，对比不同配置与量化方式下的延迟、吞吐和 GPU 显存占用。核心流程是先启动 vLLM 服务，再用自带的基准脚本与压测工具采样性能数据。

常用脚本说明：
- infra/vllm/serve.sh：启动 vLLM OpenAI 兼容服务，可用环境变量调整模型、端口、dtype 等参数。
- infra/vllm/configs/*.yaml：常见服务配置模板（FP16 / INT8 / AWQ），用于记录推荐参数。
- src/runner.py：基准客户端，读取 experiments/configs/*.yaml 发送请求并输出 CSV 结果。
- experiments/configs/baseline.yaml：基准配置示例，指定模型、提示词、请求数、输出文件等。
- experiments/prompts/sample.txt：基准提示词样例。
- bench/wrk/run_wrk.sh + bench/wrk/post.lua：wrk 压测脚本，向 /v1/completions 发送固定 JSON。
- bench/locust/run_locust.sh + bench/locust/locustfile.py：Locust 并发压测脚本，模拟多用户请求。
- tools/gpu_monitor.py：采样 nvidia-smi 输出，记录 GPU 利用率与显存占用。

KVCache_Research 子项目：
- KVCache_Research/scripts/run_benchmark.py：纯 PyTorch KV cache 研究基准入口，读取 JSON 配置并输出指标。
- KVCache_Research/configs/*.json：研究基准配置（baseline / LRU offload）。
- KVCache_Research/docs/design.md：缓存语义与指标设计说明。

## 快速开始（Ubuntu 24.04）
1) 启动 vLLM 服务
   - cd infra/vllm
   - MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" ./serve.sh

2) 运行基准客户端
   - python src/runner.py --config experiments/configs/baseline.yaml

3) 运行 wrk 压测
   - cd bench/wrk
   - URL="http://127.0.0.1:8000" ./run_wrk.sh

4) 运行 Locust 并发压测（可选）
   - cd bench/locust
   - HOST="http://127.0.0.1:8000" ./run_locust.sh

5) 记录 GPU 使用情况
   - python tools/gpu_monitor.py --out experiments/results/gpu.csv --duration 60

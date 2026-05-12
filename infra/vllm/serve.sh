#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}
PORT=${PORT:-8000}
DTYPE=${DTYPE:-"float16"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS"

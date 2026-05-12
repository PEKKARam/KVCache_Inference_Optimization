import argparse
import yaml
import torch

from kv_inference.src.vllm_benchmark import run_vllm_benchmark
from kv_inference.src.utils import load_prompts, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["benchmark"]["seed"])
    prompts = load_prompts(cfg["benchmark"]["prompts_path"])

    model_name = cfg["model"]["name"]
    dtype = cfg["model"]["dtype"]

    for batch_size in cfg["benchmark"]["batch_sizes"]:
        for in_len in cfg["benchmark"]["input_lengths"]:
            for out_len in cfg["benchmark"]["output_lengths"]:
                result = run_vllm_benchmark(
                    model_name=model_name,
                    prompts=prompts,
                    batch_size=batch_size,
                    input_len=in_len,
                    output_len=out_len,
                    dtype=dtype,
                    tensor_parallel_size=cfg["vllm"]["tensor_parallel_size"],
                    gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
                    max_num_seqs=cfg["vllm"]["max_num_seqs"],
                    enforce_eager=cfg["vllm"]["enforce_eager"],
                )
                print(result)


if __name__ == "__main__":
    main()

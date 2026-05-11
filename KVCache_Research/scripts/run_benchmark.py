from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, root_dir.as_posix())

from kvcache_research.config import load_config
from kvcache_research.model.hf_loader import load_model_and_tokenizer
from kvcache_research.benchmark.runner import run_benchmark


def _load_prompts(prompts_path: Path, num_prompts: int) -> list[str]:
    with prompts_path.open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    return prompts[:num_prompts]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KV cache benchmark")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root_dir / config_path

    cfg = load_config(config_path)
    _set_seed(cfg.benchmark.seed)

    device = cfg.model.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, tokenizer = load_model_and_tokenizer(cfg.model.name_or_path, cfg.model.dtype, device)

    prompts_path = Path(cfg.benchmark.prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = config_path.parent / prompts_path

    prompts = _load_prompts(prompts_path, cfg.benchmark.num_prompts)

    metrics = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        cache_policy=cfg.cache.policy,
        max_gpu_bytes=cfg.cache.max_gpu_bytes,
        reuse_cache=cfg.cache.reuse_cache,
        device=device,
        max_new_tokens=cfg.benchmark.max_new_tokens,
    )

    output_path = Path(cfg.benchmark.output_file)
    if not output_path.is_absolute():
        output_path = config_path.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": config_path.as_posix(),
        "device": device,
        "metrics": [item.to_dict() for item in metrics],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

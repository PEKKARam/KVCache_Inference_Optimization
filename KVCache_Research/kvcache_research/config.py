from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    name_or_path: str
    dtype: str
    device: str


@dataclass
class BenchmarkConfig:
    prompts_file: str
    num_prompts: int
    max_new_tokens: int
    seed: int
    output_file: str


@dataclass
class CacheConfig:
    policy: str
    max_gpu_bytes: int
    reuse_cache: bool


@dataclass
class RunConfig:
    model: ModelConfig
    benchmark: BenchmarkConfig
    cache: CacheConfig


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    model = ModelConfig(**raw["model"])
    benchmark = BenchmarkConfig(**raw["benchmark"])
    cache = CacheConfig(**raw["cache"])
    return RunConfig(model=model, benchmark=benchmark, cache=cache)

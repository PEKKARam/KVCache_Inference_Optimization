import time

import torch

from kv_inference.src.kvcache_demo import TinyTransformerLM
from kv_inference.src.metrics import sync_if_cuda
from kv_inference.src.utils import set_seed


def run_demo(device="cpu"):
    set_seed(42)
    model = TinyTransformerLM(vocab_size=256, dim=128, num_heads=4, num_layers=4)
    model.to(device)
    model.eval()

    prompt_len = 64
    gen_len = 64
    input_ids = torch.randint(0, model.vocab_size, (1, prompt_len), device=device)

    # Full decoding (no KV cache reuse): recompute the full prefix every step.
    start = time.perf_counter()
    seq = input_ids.clone()
    for _ in range(gen_len):
        logits, _ = model(seq)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)
    sync_if_cuda(device)
    full_time = time.perf_counter() - start

    # KV cache decoding: only process the new token each step.
    start = time.perf_counter()
    caches = None
    seq = input_ids.clone()
    _, caches = model(seq, caches)
    for _ in range(gen_len):
        next_token, caches = model.decode_step(seq[:, -1:], caches)
        seq = torch.cat([seq, next_token], dim=1)
    sync_if_cuda(device)
    cache_time = time.perf_counter() - start

    print(f"Full recompute: {full_time:.4f}s")
    print(f"With KV cache: {cache_time:.4f}s")
    if cache_time > 0:
        print(f"Speedup: {full_time / cache_time:.2f}x")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_demo(device=device)

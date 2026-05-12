import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KVCache:
    key: torch.Tensor
    value: torch.Tensor


def _reshape_for_heads(x, num_heads):
    bsz, seqlen, dim = x.shape
    head_dim = dim // num_heads
    x = x.view(bsz, seqlen, num_heads, head_dim)
    return x.transpose(1, 2)


def _merge_heads(x):
    bsz, num_heads, seqlen, head_dim = x.shape
    x = x.transpose(1, 2).contiguous()
    return x.view(bsz, seqlen, num_heads * head_dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cache=None):
        bsz, seqlen, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = _reshape_for_heads(q, self.num_heads)
        k = _reshape_for_heads(k, self.num_heads)
        v = _reshape_for_heads(v, self.num_heads)

        if cache is not None:
            k = torch.cat([cache.key, k], dim=2)
            v = torch.cat([cache.value, v], dim=2)
        new_cache = KVCache(key=k, value=v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.ones(seqlen, k.size(2), device=x.device, dtype=torch.bool)
        causal_mask = torch.tril(causal_mask)
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = _merge_heads(attn_out)
        return self.proj(attn_out), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size=256, dim=128, num_heads=4, num_layers=4, max_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, caches=None):
        bsz, seqlen = input_ids.shape
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        new_caches = []
        if caches is None:
            caches = [None] * len(self.blocks)
        for block, cache in zip(self.blocks, caches):
            x, new_cache = block(x, cache)
            new_caches.append(new_cache)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_caches

    @torch.no_grad()
    def decode_step(self, input_ids, caches=None):
        logits, caches = self.forward(input_ids, caches)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return next_token, caches

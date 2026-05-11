import torch
import time
import torch.cuda  # 添加以检查显存使用情况
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from KVCache_Research.KVM.KVCacheManager import KVCacheManager

model_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = "./model/gpt2"  # 模型目录

# 初始化模型与分词器
model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
max_length = model.config.n_positions


num_experiments = 10 # 实验次数
max_new_tokens = 50  # 最大生成 token 数

gpu_threshold = 500 * 1024 * 1024  # GPU 卸载阈值
off_threshold = 5  # 卸载低利用率缓存的阈值
pre_threshold = 10 # 预取高利用率缓存的阈值
step_interval = 10  # 每隔多少步检查一次显存使用情况

# 初始化 KVCache 管理器
kvcache_manager = KVCacheManager(device=device)


def infer_with_kvm(text):
    """
    推理过程：支持动态卸载 + 预取的 KV 缓存管理。
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_key_values = None

    start_time = time.time()
    ttft = None
    total_token_time = 0
    memory_usage = []
    peak_memory_usage = 0

    for step in range(max_new_tokens):
        step_start_time = time.time()

        # === 从缓存中读取历史 past_key_values（每层每步） ===
        if step > 0:
            past_key_values = []
            for layer in range(model.config.n_layer):
                try:
                    key = f"layer_{layer}_step_{step - 1}_key"
                    kv = kvcache_manager.get_cache(key)
                except KeyError:
                    kv = None
                past_key_values.append(kv)
        else:
            past_key_values = None

        # === 模型推理 ===
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # === 存入当前步的 KVCache ===
        for layer, (k, v) in enumerate(past_key_values):
            kvcache_manager.add_cache(
                key=f"layer_{layer}_step_{step}_key",
                value=(k, v),
                to_gpu=True
            )

        # === GPU 显存动态检查 ===
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(device)
            mem_reserved = torch.cuda.memory_reserved(device)
            mem_free = mem_reserved - mem_allocated
            memory_usage.append(mem_allocated)
            peak_memory_usage = max(peak_memory_usage, mem_allocated)

            # 若显存紧张，触发缓存卸载（低频移至 CPU）
            if mem_free < gpu_threshold:
                kvcache_manager.offload_cache(threshold=off_threshold)

        # === 每 step_interval 步动态管理缓存（预取/卸载）===
        if step > 0 and step % step_interval == 0:
            kvcache_manager.manage_cache(
                offload_threshold=off_threshold,
                prefetch_threshold=pre_threshold
            )

        # === 贪心解码下一 token ===
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = next_token
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        # === 首 token 生成时间 ===
        if ttft is None:
            ttft = time.time() - start_time
        total_token_time += time.time() - step_start_time

    end_time = time.time()
    tbt = end_time - start_time
    tpot = total_token_time / max_new_tokens
    avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0

    return {
        "TTFT": ttft,
        "TPOT": tpot,
        "TBT": tbt,
        "TBT_Total": tbt,
        "Avg_Memory_Usage": avg_memory / (1024 * 1024),
        "Peak_Memory_Usage": peak_memory_usage / (1024 * 1024)
    }

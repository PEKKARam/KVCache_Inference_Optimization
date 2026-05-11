import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2-xl" 
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "./model/gpt2-xl"  # 模型目录

# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model.save_pretrained(model_dir)  # 保存模型
# tokenizer.save_pretrained(model_dir)  # 保存分词器

# 初始化模型与分词器
model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
model.eval()
max_length = model.config.n_positions

# 实验次数
num_experiments = 10
max_new_tokens = 50  # 最大生成 token 数

gpu_threshold = 500 * 1024 * 1024  # GPU 卸载阈值
off_threshold = 5  # 卸载低利用率缓存的阈值
pre_threshold = 10 # 预取高利用率缓存的阈值
step_interval = 10  # 每隔多少步检查一次显存使用情况

def infer_baseline(text):
    """
    对单条输入文本进行推理，baseline实验。
    :param text: 输入文本
    :return: TTFT、TPOT、 TBT_Total、Avg_Memory_Usage、Peak_Memory_Usage
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_key_values = None

    start_time = time.time()
    ttft = None  # 初始化 TTFT
    total_token_time = 0  # 累计每个 token 的生成时间

    memory_usage = []
    peak_memory_usage = 0

    for step in range(max_new_tokens):
        step_start_time = time.time()
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values  # 更新 KVCache

        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated(device)

            # 记录当前显存占用
            memory_usage.append(gpu_memory_allocated)
            peak_memory_usage = max(peak_memory_usage, gpu_memory_allocated)

        # 获取下一个 token（贪心搜索）
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = next_token
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        # 记录 TTFT（首次生成 token 的时间）
        if ttft is None:
            ttft = time.time() - start_time

        total_token_time += time.time() - step_start_time

    end_time = time.time()
    tbt = end_time - start_time  # 总推理时间
    tpot = total_token_time / max_new_tokens  # 每个 token 的平均生成时间

    # 计算平均显存占用
    avg_memory_usage = sum(memory_usage) / len(memory_usage) if memory_usage else 0

    return {
        "TTFT": ttft,
        "TPOT": tpot,
        "TBT": tbt,
        "TBT_Total": tbt,
        "Avg_Memory_Usage": avg_memory_usage / (1024 * 1024),  # 转换为 MB
        "Peak_Memory_Usage": peak_memory_usage / (1024 * 1024)  # 转换为 MB
    }
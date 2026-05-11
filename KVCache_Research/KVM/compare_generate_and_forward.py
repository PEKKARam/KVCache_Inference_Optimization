import torch
import time
from KVCache_Research.Baseline.config import model, tokenizer, device, max_length

# 测试文本
text = "The quick brown fox"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

# 使用 forward 方法
def test_forward():
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    past_key_values = None
    generated_tokens = []

    start_time = time.time()
    for _ in range(50):  # 生成 50 个 token
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # 获取下一个 token（贪心搜索）
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())

        # 更新输入
        input_ids = next_token
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
    end_time = time.time()

    print(f"使用 forward 方法生成的文本: {tokenizer.decode(generated_tokens)}")
    print(f"使用 forward 方法的推理时间: {end_time - start_time:.4f} 秒")

# 使用 generate 方法
def test_generate():
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id
    )
    end_time = time.time()

    print(f"使用 generate 方法生成的文本: {tokenizer.decode(outputs[0])}")
    print(f"使用 generate 方法的推理时间: {end_time - start_time:.4f} 秒")

# 测试两种方法
test_forward()
test_generate()
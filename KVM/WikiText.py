import numpy as np
from tqdm import tqdm
from utils import get_dataset
from config import model_name, device, num_experiments, infer_with_kvm

# 定义本地数据集路径
local_dataset_path = "./datasets/wikitext-2"

# 加载测试集
dataset = get_dataset("wikitext-2", local_dataset_path, subset="wikitext-2-raw-v1")
texts = dataset["test"]["text"][:100]  # 选取前100条测试样本

# 存储每次实验的平均延迟指标和总推理时间
experiment_metrics = []
experiment_times = []
experiment_memory_usage = []  # 存储显存占用指标

for experiment in range(num_experiments):
    print(f"开始第 {experiment + 1}/{num_experiments} 次实验 ...")
    ttft_list, tpot_list, tbt_list = [], [], []
    inference_times = []
    avg_memory_list, peak_memory_list = [], []  # 存储每次推理的显存占用

    for text in tqdm(texts, desc=f"实验 {experiment + 1}/{num_experiments}", unit="样本"):
        if len(text.strip()) > 0:
            metrics = infer_with_kvm(text)
            inference_times.append(metrics["TBT_Total"])

            ttft_list.append(metrics["TTFT"])
            tpot_list.append(metrics["TPOT"])
            tbt_list.append(metrics["TBT"])

            avg_memory_list.append(metrics["Avg_Memory_Usage"])
            peak_memory_list.append(metrics["Peak_Memory_Usage"])


    # 计算每次实验的平均延迟指标
    avg_ttft = np.mean(ttft_list)
    avg_tpot = np.mean(tpot_list)
    avg_tbt = np.mean(tbt_list)
    experiment_metrics.append({"TTFT": avg_ttft, "TPOT": avg_tpot, "TBT": avg_tbt})

    # 计算每次实验的平均推理时间
    avg_inference_time = np.mean(inference_times)
    experiment_times.append(avg_inference_time)

    # 新增：计算显存占用的平均值和峰值
    avg_memory_usage = np.mean(avg_memory_list)
    peak_memory_usage = np.max(peak_memory_list)
    experiment_memory_usage.append({"Avg_Memory_Usage": avg_memory_usage, "Peak_Memory_Usage": peak_memory_usage})

    print(f"第 {experiment + 1} 次实验 平均 TTFT：{avg_ttft:.4f} 秒")
    print(f"第 {experiment + 1} 次实验 平均 TPOT：{avg_tpot:.4f} 秒")
    print(f"第 {experiment + 1} 次实验 平均 TBT：{avg_tbt:.4f} 秒")
    print(f"第 {experiment + 1} 次实验 平均推理时间：{avg_inference_time:.4f} 秒")
    print(f"第 {experiment + 1} 次实验 平均显存占用：{avg_memory_usage:.2f} MB")
    print(f"第 {experiment + 1} 次实验  峰值显存占用：{peak_memory_usage:.2f} MB")

# 计算总体平均值和标准差
overall_metrics = {
    "TTFT": np.mean([m["TTFT"] for m in experiment_metrics]),
    "TPOT": np.mean([m["TPOT"] for m in experiment_metrics]),
    "TBT": np.mean([m["TBT"] for m in experiment_metrics]),
}
std_metrics = {
    "TTFT": np.std([m["TTFT"] for m in experiment_metrics]),
    "TPOT": np.std([m["TPOT"] for m in experiment_metrics]),
    "TBT": np.std([m["TBT"] for m in experiment_metrics]),
}
overall_avg_time = np.mean(experiment_times)
std_dev_time = np.std(experiment_times)

# 新增：计算显存占用的总体平均值和峰值
overall_memory_usage = {
    "Avg_Memory_Usage": np.mean([m["Avg_Memory_Usage"] for m in experiment_memory_usage]),
    "Peak_Memory_Usage": np.max([m["Peak_Memory_Usage"] for m in experiment_memory_usage]),
}

print(f"设备: {device}\n模型: {model_name}\n数据集: {local_dataset_path}")
print(f"采用KV缓存卸载优化 - 平均 TTFT：{overall_metrics['TTFT']:.4f} 秒 (标准差: {std_metrics['TTFT']:.4f})")
print(f"采用KV缓存卸载优化 - 平均 TPOT：{overall_metrics['TPOT']:.4f} 秒 (标准差: {std_metrics['TPOT']:.4f})")
print(f"采用KV缓存卸载优化 - 平均 TBT：{overall_metrics['TBT']:.4f} 秒 (标准差: {std_metrics['TBT']:.4f})")
print(f"采用KV缓存卸载优化 - 总体平均推理时间：{overall_avg_time:.4f} 秒 (标准差: {std_dev_time:.4f})")
print(f"采用KV缓存卸载优化 - 平均显存占用：{overall_memory_usage['Avg_Memory_Usage']:.2f} MB")
print(f"采用KV缓存卸载优化 - 峰值显存占用：{overall_memory_usage['Peak_Memory_Usage']:.2f} MB")
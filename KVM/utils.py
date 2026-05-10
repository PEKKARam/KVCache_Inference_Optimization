import os
from datasets import load_dataset, Dataset, DatasetDict

def get_dataset(dataset_name, local_dataset_path, subset=None, split=None):
    """
    获取数据集。如果本地存在缓存，则从本地加载；否则从 Hugging Face 下载并保存到本地。
    
    :param dataset_name: str, 数据集名称（如 "wikitext"）
    :param local_dataset_path: str, 本地保存数据集的路径
    :param subset: str, 数据集的子集名称（如 "wikitext-2-raw-v1"），默认为 None
    :param split: str, 数据集的分割（如 "train", "test", "validation"），默认为 None
    :return: Dataset 或 DatasetDict 对象
    """
    if local_dataset_path is None:
        raise ValueError("local_dataset_path cannot be None")
    
    # 检查本地路径是否存在
    if not os.path.exists(local_dataset_path):
        print(f"正在下载数据集 {dataset_name}...")
        # 下载数据集
        dataset = load_dataset(dataset_name, subset, split=split)
        os.makedirs(os.path.dirname(local_dataset_path), exist_ok=True)  # 确保目录存在
        dataset.save_to_disk(local_dataset_path)  # 保存到本地
    else:
        print(f"从本地加载数据集 {dataset_name}...")
        # 检查是 Dataset 还是 DatasetDict
        if split is not None:
            dataset = Dataset.load_from_disk(local_dataset_path)  # 加载单一分割
        else:
            dataset = DatasetDict.load_from_disk(local_dataset_path)  # 加载完整数据集
    
    return dataset
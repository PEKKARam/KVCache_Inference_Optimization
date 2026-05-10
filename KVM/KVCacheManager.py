import torch
import heapq

class KVCacheManager:
    def __init__(self, device="cuda"):
        """
        初始化 KVCache 管理器
        :param device: 默认设备(如 "cuda" 或 "cpu")
        """
        self.device = device
        self.cache = {}  # 存储 KVCache 的字典
        self.access_count = {}  # 记录每个缓存项的访问次数
        self.min_heap = []  # 最小堆，用于维护访问频率最高的缓存项
        self.key_to_heap_index = {}  # 记录每个键在堆中的索引
        print(f"KVCacheManager 初始化，默认设备: {device}")

    def add_cache(self, key, value, to_gpu=True):
        """
        添加 KVCache
        :param key: 缓存的键
        :param value: 缓存的值（张量或元组）
        :param to_gpu: 是否将缓存存储在 GPU 上
        """
        if isinstance(value, tuple):  # 如果是元组，逐个张量移动到目标设备
            if to_gpu and torch.cuda.is_available():
                self.cache[key] = tuple(v.to("cuda") if isinstance(v, torch.Tensor) else v for v in value)
            else:
                self.cache[key] = tuple(v.to("cpu") if isinstance(v, torch.Tensor) else v for v in value)
        else:  # 如果是单个张量
            if to_gpu and torch.cuda.is_available():
                self.cache[key] = value.to("cuda")
            else:
                self.cache[key] = value.to("cpu")
        self.access_count[key] = 0  # 初始化访问计数
        heapq.heappush(self.min_heap, (0, key))  # 初始化堆中对应项
        self.key_to_heap_index[key] = len(self.min_heap) - 1

    def get_cache(self, key):
        """
        获取 KVCache
        :param key: 缓存的键
        :return: 缓存的值（张量或元组）
        """
        if key in self.cache:
            self.access_count[key] += 1  # 增加访问计数
            heapq.heappush(self.min_heap, (-self.access_count[key], key))  # 更新堆
            self.key_to_heap_index[key] = len(self.min_heap) - 1
            return self.cache[key]
        else:
            raise KeyError(f"缓存中不存在键: {key}")

    def move_to_cpu(self, key):
        """
        将指定的 KVCache 移动到 CPU
        :param key: 缓存的键
        """
        if key in self.cache:
            if isinstance(self.cache[key], tuple):  # 如果是元组
                self.cache[key] = tuple(v.to("cpu") if isinstance(v, torch.Tensor) else v for v in self.cache[key])
            else:  # 如果是单个张量
                self.cache[key] = self.cache[key].to("cpu")
            # print(f"将缓存 {key} 移动到 CPU")
        else:
            raise KeyError(f"缓存中不存在键: {key}")

    def move_to_gpu(self, key):
        """
        将指定的 KVCache 移动到 GPU
        :param key: 缓存的键
        """
        if key in self.cache:
            if isinstance(self.cache[key], tuple):  # 如果是元组
                self.cache[key] = tuple(v.to("cuda") if isinstance(v, torch.Tensor) else v for v in self.cache[key])
            else:  # 如果是单个张量
                self.cache[key] = self.cache[key].to("cuda")
            # print(f"将缓存 {key} 移动到 GPU")
        else:
            raise KeyError(f"缓存中不存在键: {key}")

    def clear_cache(self, key=None):
        """
        清理缓存
        :param key: 如果指定键，则清理该键对应的缓存；否则清理所有缓存
        """
        if key:
            if key in self.cache:
                del self.cache[key]
                del self.access_count[key]
            if key in self.key_to_heap_index:
                del self.key_to_heap_index[key]
        else:
            self.cache.clear()
            self.access_count.clear()
            self.min_heap.clear()
            self.key_to_heap_index.clear()

    def evaluate_cache_usage(self, top_n=5):
        """
        返回访问频率最高的前 N 个缓存项
        :param top_n: 返回的缓存项数量
        :return: 利用率最高的缓存项列表 [(key, access_count), ...]
        """
        # 使用 sorted 对访问计数进行排序
        sorted_cache = sorted(self.access_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_cache[:top_n]

    def evaluate_least_used_cache(self, bottom_n=5):
        """
        返回访问频率最低的前 N 个缓存项
        :param bottom_n: 返回的缓存项数量
        :return: 利用率最低的缓存项列表 [(key, access_count), ...]
        """
        # 使用 sorted 对访问计数进行排序
        sorted_cache = sorted(self.access_count.items(), key=lambda x: x[1])
        return sorted_cache[:bottom_n]

    def offload_cache(self, threshold=5):
        """
        将访问频率低于阈值的缓存从 GPU 移动到 CPU
        :param threshold: 访问频率阈值
        """
        for key, access_count in self.access_count.items():
            if access_count < threshold and key in self.cache:
                self.move_to_cpu(key)
                # print(f"缓存 {key} 已从 GPU 移动到 CPU，访问计数: {access_count}")

    def prefetch_cache(self, threshold=10):
        """
        将访问频率高于阈值的缓存从 CPU 移动到 GPU
        :param threshold: 访问频率阈值
        """
        for key, access_count in self.access_count.items():
            if access_count >= threshold and key in self.cache:
                self.move_to_gpu(key)
                # print(f"缓存 {key} 已从 CPU 移动到 GPU，访问计数: {access_count}")

    def manage_cache(self, offload_threshold=5, prefetch_threshold=10):
        """
        动态管理缓存，根据访问频率在 CPU 和 GPU 之间移动缓存
        :param offload_threshold: GPU -> CPU 的访问频率阈值
        :param prefetch_threshold: CPU -> GPU 的访问频率阈值
        """
        self.offload_cache(threshold=offload_threshold)
        self.prefetch_cache(threshold=prefetch_threshold)

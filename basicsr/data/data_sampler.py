import math
import torch
from torch.utils.data.sampler import Sampler

class EnlargedSampler(Sampler):
    """
    一个用于数据加载的采样器，它限制了对数据集的子集进行加载，并支持扩大数据集以便于基于迭代的训练。
    
    这个类是对 torch.utils.data.distributed.DistributedSampler 的修改，
    支持扩大数据集，以节省在每个epoch结束后重启数据加载器的时间。

    参数:
        dataset (torch.utils.data.Dataset): 用于采样的数据集。
        num_replicas (int | None): 参与训练的进程数量。通常是world_size。
        rank (int | None): 当前进程在num_replicas中的排名。
        ratio (int): 数据集扩大的比例。默认为1，即不扩大。
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        """
        初始化采样器对象。
        
        参数:
            dataset: 用于采样的数据集。
            num_replicas: 参与训练的进程数量。
            rank: 当前进程的排名。
            ratio: 数据集扩大的比例。
        """
        self.dataset = dataset  # 数据集对象
        self.num_replicas = num_replicas  # 参与训练的总进程数
        self.rank = rank  # 当前进程的排名
        self.epoch = 0  # 当前的epoch数，用于确定性洗牌
        self.num_samples = math.ceil(
            len(self.dataset) * ratio / self.num_replicas)  # 每个进程需要采样的数量
        self.total_size = self.num_samples * self.num_replicas  # 扩大后的总样本数

    def __iter__(self):
        """
        创建一个迭代器，用于生成数据采样的索引。
        
        返回:
            一个迭代器，包含当前进程应该访问的数据索引。
        """
        g = torch.Generator()  # 创建一个新的随机数生成器
        g.manual_seed(self.epoch)  # 使用当前的epoch数作为随机种子
        # 生成一个随机排列的索引列表
        indices = torch.randperm(self.total_size, generator=g).tolist()

        # 获取原始数据集的大小
        dataset_size = len(self.dataset)
        # 将超出数据集大小的索引转换为数据集范围内的索引
        indices = [v % dataset_size for v in indices]

        # 从洗牌后的索引中选取当前进程的子集
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples  # 确保索引数量正确

        return iter(indices)  # 返回索引的迭代器

    def __len__(self):
        """
        返回每个进程中采样的样本数量。
        
        返回:
            每个进程中采样的样本数量。
        """
        return self.num_samples

    def set_epoch(self, epoch):
        """
        设置当前的epoch数，用于在迭代过程中确定性洗牌。

        参数:
            epoch (int): 当前的epoch数。
        """
        self.epoch = epoch  # 更新当前的epoch数

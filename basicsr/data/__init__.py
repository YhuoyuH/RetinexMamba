import importlib
import numpy as np
import random
import torch
import torch.utils.data
from functools import partial
from os import path as osp
from pdb import set_trace as stx

from basicsr.data.prefetch_dataloader import PrefetchDataLoader
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.dist_util import get_dist_info

__all__ = ['create_dataset', 'create_dataloader']  # 定义模块公开接口

# 自动扫描并导入数据集模块
# 扫描data文件夹下所有文件名中包含'_dataset'的文件
data_folder = osp.dirname(osp.abspath(__file__))  # 获取当前文件所在目录
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder)
    if v.endswith('_dataset.py')  # 选择以'_dataset.py'结尾的文件
]

# 导入所有以'_dataset'结尾的模块
_dataset_modules = [
    importlib.import_module(f'basicsr.data.{file_name}')  # 动态导入模块
    for file_name in dataset_filenames
]

def create_dataset(dataset_opt):

    """
    创建数据集对象。

    根据配置字典（dataset_opt）中的参数动态创建数据集实例。它会查找与配置中指定的数据集类型（'type'）匹配的类，并实例化它。

    参数:
        dataset_opt (dict): 包含数据集配置信息的字典，如数据集名称和类型。

    返回:
        数据集对象实例。
    """

    dataset_type = dataset_opt['type']  # 获取数据集类型

    # 动态实例化，逐个检查并尝试创建数据集实例
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)  # 尝试获取对应的数据集类
        if dataset_cls is not None:
            break  # 如果找到了相应的类，就中断循环
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')  # 如果没有找到类，抛出异常

    dataset = dataset_cls(dataset_opt)  # 实例化数据集

    logger = get_root_logger()  # 获取日志记录器
    logger.info(
        f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} '
        'is created.')  # 记录日志
    return dataset  # 返回数据集实例

def create_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):

    """
    创建数据加载器对象。

    根据给定的数据集、配置选项以及训练环境设置（如是否分布式训练、GPU数量等）创建数据加载器。
    这个函数处理不同训练阶段（训练、验证、测试）的数据加载逻辑，并且可以配置数据预取机制来优化加载性能。

    参数:
        dataset (torch.utils.data.Dataset): 要加载的数据集对象。
        dataset_opt (dict): 包含数据加载器配置信息的字典。
        num_gpu (int): 使用的GPU数量，默认为1。
        dist (bool): 是否进行分布式训练，默认为False。
        sampler (torch.utils.data.sampler): 数据采样器，默认为None。
        seed (int | None): 随机种子，默认为None。

    返回:
        数据加载器对象实例。
    """

    phase = dataset_opt['phase']  # 获取数据集阶段（训练、验证、测试）
    rank, _ = get_dist_info()  # 获取分布式训练信息

    if phase == 'train':  # 训练阶段
        if dist:  # 分布式训练
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # 非分布式训练
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # 默认不打乱，除非后面设置为True
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)  # 在训练时，为了保证批次的一致性，通常会丢弃最后一个不完整的批次
        if sampler is None:
            dataloader_args['shuffle'] = True  # 如果没有指定采样器，就打乱数据
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank,
            seed=seed) if seed is not None else None  # 初始化工作线程
    elif phase in ['val', 'test']:  # 验证或测试阶段
        dataloader_args = dict(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. '
                         "Supported ones are 'train', 'val' and 'test'.")  # 阶段不正确时抛出异常

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)  # 设置是否钉住内存

    prefetch_mode = dataset_opt.get('prefetch_mode')  # 获取预取模式
    if prefetch_mode == 'cpu':  # CPU预取
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: '
                    f'num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: 普通数据加载器
        # prefetch_mode='cuda': 用于CUDA预取的数据加载器
        return torch.utils.data.DataLoader(**dataloader_args)

def worker_init_fn(worker_id, num_workers, rank, seed):

    """
    初始化数据加载器工作线程的函数。

    设置每个工作线程的随机种子，以确保数据加载过程的可重复性。这对于实验的可复现性和避免过拟合等问题至关重要。

    参数:
        worker_id (int): 当前工作线程的ID。
        num_workers (int): 总的工作线程数量。
        rank (int): 当前进程在分布式训练中的等级或编号。
        seed (int): 基础随机种子。

    返回:
        无返回值，直接对工作线程的状态进行设置。
    """
    worker_seed = num_workers * rank + worker_id + seed  # 计算工作线程的随机种子
    np.random.seed(worker_seed)  # 设置NumPy的随机种子
    random.seed(worker_seed)  # 设置Python内置random模块的随机种子

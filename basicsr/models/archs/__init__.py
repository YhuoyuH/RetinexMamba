import importlib
from os import path as osp
from pdb import set_trace as stx  # 引入pdb调试工具的设置断点函数

from basicsr.utils import scandir  # 从basicsr.utils导入scandir函数

# 自动扫描并导入架构模块
# 扫描'archs'文件夹下的所有文件，并收集以'_arch.py'结尾的文件
arch_folder = osp.dirname(osp.abspath(__file__))  # 获取当前文件所在目录的绝对路径
arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
    if v.endswith('_arch.py')  # 筛选出以'_arch.py'结尾的文件
]

# 导入所有架构模块
_arch_modules = [
    importlib.import_module(f'basicsr.models.archs.{file_name}')  # 动态导入模块
    for file_name in arch_filenames
]

# stx()  # 调试时启用断点

def dynamic_instantiation(modules, cls_type, opt):
    """
    动态实例化类。

    参数:
        modules (list[importlib modules]): importlib模块列表。
        cls_type (str): 类型。
        opt (dict): 类初始化的关键字参数。

    返回:
        class: 实例化的类。
    """
    cls_ = None
    for module in modules:  # 遍历模块列表
        cls_ = getattr(module, cls_type, None)  # 尝试获取类类型
        if cls_ is not None:  # 如果找到了类，则跳出循环
            break
    if cls_ is None:  # 如果没有找到类，则抛出异常
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)  # 实例化并返回类

def define_network(opt):
    """
    根据选项定义网络。

    参数:
        opt (dict): 包含网络配置的字典。

    返回:
        net: 实例化的网络对象。
    """
    network_type = opt.pop('type')  # 获取网络类型并从选项中移除
    net = dynamic_instantiation(_arch_modules, network_type, opt)  # 动态实例化网络
    return net  # 返回网络实例

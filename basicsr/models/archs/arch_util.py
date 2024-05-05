import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.utils import get_root_logger

# 初始化模块的权重
@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """
    初始化网络权重。

    参数:
        module_list (list[nn.Module] | nn.Module): 需要初始化的模块或模块列表。
        scale (float): 权重的缩放因子，特别用于残差块。默认为1。
        bias_fill (float): 偏置项填充值。默认为0。
        kwargs (dict): 初始化函数的其他参数。
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale  # 权重缩放
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)  # 偏置填充
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale  # 权重缩放
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)  # 偏置填充
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)  # BN层权重初始化
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)  # BN层偏置填充

# 通过堆叠相同的基础块来构建层
def make_layer(basic_block, num_basic_block, **kwarg):
    """
    通过堆叠相同的基础块来构造层。

    参数:
        basic_block (nn.module): 基础块的类。
        num_basic_block (int): 基础块的数量。

    返回:
        nn.Sequential: 一个由基础块组成的层。
    """
    layers = [basic_block(**kwarg) for _ in range(num_basic_block)]
    return nn.Sequential(*layers)

# 定义不带BN的残差块
class ResidualBlockNoBN(nn.Module):
    """
    不带批量归一化的残差块。
    结构为: ---Conv-ReLU-Conv-+-
             |________________|

    参数:
        num_feat (int): 中间特征的通道数，默认为64。
        res_scale (float): 残差缩放系数，默认为1。
        pytorch_init (bool): 是否使用PyTorch默认初始化方法，默认为False。
    """
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

# 定义上采样模块
class Upsample(nn.Sequential):
    """
    上采样模块。

    参数:
        scale (int): 放大因子，支持2^n和3。
        num_feat (int): 中间特征的通道数。
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # 判断scale是否为2的幂
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError('不支持的放大因子。支持的放大因子为2^n和3。')
        super(Upsample, self).__init__(*m)

# 光流扭曲
def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    使用光流扭曲图像或特征图。

    参数:
        x (Tensor): 形状为(n, c, h, w)的张量。
        flow (Tensor): 形状为(n, h, w, 2)的张量，表示光流。
        interp_mode (str): 插值模式，'nearest'或'bilinear'。默认为'bilinear'。
        padding_mode (str): 填充模式，'zeros'、'border'或'reflection'。默认为'zeros'。
        align_corners (bool): 是否对齐角点。默认为True。

    返回:
        Tensor: 扭曲后的图像或特征图。
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False

    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
    return output

# 调整光流大小
def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """
    根据比例或形状调整光流的大小。

    参数:
        flow (Tensor): 预计算的光流，形状为[N, 2, H, W]。
        size_type (str): 'ratio'或'shape'。
        sizes (list[int | float]): 缩放比例或输出形状。
        interp_mode (str): 插值模式，默认为'bilinear'。
        align_corners (bool): 是否对齐角点，默认为False。

    返回:
        Tensor: 调整大小后的光流。
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError('size_type应为ratio或shape。')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow

# 像素逆洗牌
def pixel_unshuffle(x, scale):
    """
    像素逆洗牌。

    参数:
        x (Tensor): 输入特征，形状为(b, c, hh, hw)。
        scale (int): 下采样比率。

    返回:
        Tensor: 像素逆洗牌后的特征。
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# class DCNv2Pack(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     Ref:
#         Delving Deep into Deformable Alignment in Video Super-Resolution.
#     """
#
#     def forward(self, x, feat):
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(
#                 f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)

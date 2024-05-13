# export PYTHONPATH= $PYTHONPATH:/mnt/workspace/RetinexMamba/basicsr/models/archs/vmamba_arch.py

#将IGMSA替换为IFA进行特征融合
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from .SS2D_arch import SS2D
from .IFA_arch import IFA
import sys
print(sys.path)
# 这两个函数 _no_grad_trunc_normal_ 和 trunc_normal_ 都用于初始化神经网络中的权重参数。它们将权重初始化为遵循截断正态分布的值，这种方法有助于在训练开始时改善网络的性能和稳定性。
def _no_grad_trunc_normal_(tensor, mean, std, a, b):#这是一个内部函数，用于实际执行初始化过程。
    """
    用截断正态分布初始化张量。

    在给定的范围 [a, b] 内，根据指定的均值 (mean) 和标准差 (std) 截断正态分布，
    并用这个分布来填充输入的张
    量。
    

    参数:
        tensor (Tensor): 需要被初始化的张量。
        mean (float): 截断正态分布的均值。
        std (float): 截断正态分布的标准差。
        a (float): 分布的下限。
        b (float): 分布的上限。

    返回:
        Tensor: 填充后的张量，其值遵循指定的截断正态分布。
    
    功能：
        这个函数主要用于深度学习中的权重初始化，尤其是在需要限制权重范围以避免激活值过大或过小时非常有用。通过截断正态分布进行初始化，可以帮助神经网络更快地收敛并提高模型的稳定性。
    """
    # 计算正态分布的累积分布函数（CDF）值
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # 检查均值是否在截断区间外的两个标准差范围内
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():  # 确保以下操作不计算梯度
        # 计算截断区间的累积分布函数值
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)
        # 以这个区间初始化张量
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()  # 计算逆误差函数，以获得正态分布的值
        tensor.mul_(std * math.sqrt(2.))  # 缩放
        tensor.add_(mean)  # 加上均值
        tensor.clamp_(min=a, max=b)  # 截断到 [a, b]

        return tensor  # 返回初始化后的张量

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):#这是一个公开的接口函数，通常由外部调用来初始化权重。
    """
    将输入张量初始化为截断正态分布。

    这个函数是`_no_grad_trunc_normal_`的公开接口，用于初始化张量，使其值遵循指定均值和标准差的截断正态分布。

    参数:
        tensor (torch.Tensor): 需要初始化的张量。
        mean (float, 可选): 正态分布的均值，默认为 0。
        std (float, 可选): 正态分布的标准差，默认为 1。
        a (float, 可选): 分布的下限，默认为 -2。
        b (float, 可选): 分布的上限，默认为 2。

    返回:
        torch.Tensor: 初始化后的张量，其值遵循指定的截断正态分布。
    
    功能：
        这个函数通常用于神经网络的权重初始化过程中，特别是当我们需要确保权重在特定范围内且遵循正态分布时。通过截断分布，我们可以避免极端值的影响，从而帮助提高模型训练的稳定性和效率。
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    """
    对张量进行变量缩放初始化。

    这个函数根据张量的维度(fan_in, fan_out)和给定的参数，对张量进行初始化，
    以确保初始化的张量具有一定的方差，这有助于控制网络层输出的方差在训练初期保持稳定。

    参数:
        tensor (torch.Tensor): 需要初始化的张量。
        scale (float): 缩放因子,用于调整方差,默认为1.0。
        mode (str): 指定使用张量的哪部分维度来计算方差。
                    'fan_in'：使用输入维度，
                    'fan_out'：使用输出维度，
                    'fan_avg'：使用输入和输出维度的平均值。
                    默认为'fan_in'。
        distribution (str): 初始化的分布类型，可以是
                    'truncated_normal'：截断正态分布，
                    'normal'：正态分布，
                    'uniform'：均匀分布。
                    默认为'normal'。

    返回:
        None，直接在输入的张量上进行修改。
    
    功能：
        这种初始化方法通常用于深度神经网络中，可以帮助保持激活函数输入的方差在训练过程中保持稳定，从而有助于避免梯度消失或爆炸问题。通过调整scale、mode和distribution参数，可以进一步优化模型的初始化过程。
    """

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)  # 计算张量的输入和输出维度大小 

    # 根据mode确定分母
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom  # 计算方差

    # 根据指定的分布进行初始化
    if distribution == "truncated_normal":
        # 使用截断正态分布进行初始化，std是标准差
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        # 使用正态分布进行初始化
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        # 使用均匀分布进行初始化
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        # 如果提供了无效的分布类型，抛出异常
        raise ValueError(f"invalid distribution {distribution}")

def lecun_normal_(tensor):
    """
    使用 LeCun 正态初始化方法对张量进行初始化。

    LeCun 初始化是一种变量缩放方法，特别适用于带有S型激活函数（如sigmoid或tanh）的深度学习模型。
    它根据层的输入数量（fan_in）来调整权重的缩放，从而使网络的训练更加稳定。

    参数:
        tensor (torch.Tensor): 需要初始化的张量。

    返回:
        None，直接在输入的张量上进行修改。
    """
    # 调用 variance_scaling_ 函数，使用'fan_in'模式和'normal'分布进行初始化
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

class PreNorm(nn.Module):
    """
    预归一化模块，通常用于Transformer架构中。

    在执行具体的功能（如自注意力或前馈网络）之前先进行层归一化，
    这有助于稳定训练过程并提高模型性能。

    属性:
        dim: 输入特征的维度。
        fn: 要在归一化后应用的模块或函数。
    """

    def __init__(self, dim, fn):
        """
        初始化预归一化模块。

        参数:
            dim (int): 输入特征的维度，也是层归一化的维度。
            fn (callable): 在归一化之后应用的模块或函数。
        """
        super().__init__()  # 初始化基类 nn.Module
        self.fn = fn  # 存储要应用的函数或模块
        self.norm = nn.LayerNorm(dim)  # 创建层归一化模块

    def forward(self, x, *args, **kwargs):
        """
        对输入数据进行前向传播。

        参数:
            x (Tensor): 输入到模块的数据。
            *args, **kwargs: 传递给self.fn的额外参数。

        返回:
            Tensor: self.fn的输出，其输入是归一化后的x。
        """
        x = self.norm(x)  # 首先对输入x进行层归一化
        return self.fn(x, *args, **kwargs)  # 将归一化的数据传递给self.fn，并执行

class GELU(nn.Module):
    """
    GELU激活函数的封装。

    GELU (Gaussian Error Linear Unit) 是一种非线性激活函数，
    它被广泛用于自然语言处理和深度学习中的其他领域。
    这个函数结合了ReLU和正态分布的性质。
    """

    def forward(self, x):
        """
        在输入数据上应用GELU激活函数。

        参数:
            x (Tensor): 输入到激活函数的数据。

        返回:
            Tensor: 经过GELU激活函数处理后的数据。
        """
        return F.gelu(x)  # 使用PyTorch的函数实现GELU激活

def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    """
    创建并返回一个二维卷积层。

    参数:
    in_channels (int): 输入特征图的通道数。
    out_channels (int): 输出特征图的通道数。
    kernel_size (int): 卷积核的大小，卷积核是正方形的。
    bias (bool, 可选): 是否在卷积层中添加偏置项。默认为 False。
    padding (int, 可选): 在输入特征图周围添加的零填充的层数。默认为 1。
    stride (int, 可选): 卷积的步长。默认为 1。

    返回:
    nn.Conv2d: 配置好的二维卷积层。
    """
    # 创建并返回一个二维卷积层，具体配置参数包括输入通道数、输出通道数、卷积核大小等。
    # padding 参数设置为卷积核大小的一半，确保输出特征图的大小与步长和填充有关，但如果步长为 1，则与输入相同。
    return nn.Conv2d(
        in_channels,  # 输入特征的通道数
        out_channels,  # 输出特征的通道数
        kernel_size,  # 卷积核的大小
        padding=(kernel_size // 2),  # 自动计算填充大小以保持特征图的空间尺寸
        bias=bias,  # 是否添加偏置项
        stride=stride  # 卷积的步长
    )

def shift_back(inputs, step=2):
    """
    对输入张量进行列向移动裁剪，以使输出张量的列数减少。

    参数:
    inputs (Tensor): 输入的四维张量，形状为 [batch_size, channels, rows, cols]。
    step (int, 可选): 每个通道移动的基础步长，默认为2。

    返回:
    Tensor: 列数被裁剪的输出张量，形状为 [batch_size, channels, rows, 256]。
    """
    [bs, nC, row, col] = inputs.shape  # 提取输入张量的形状维度

    # 计算下采样比率，此处特定用途为256固定列输出
    down_sample = 256 // row

    # 调整步长以反映下采样比率的平方影响
    step = float(step) / float(down_sample * down_sample)

    # 设置输出列数，此处为输入行数，这里假设行数等于期望的列数
    out_col = row

    # 遍历每个通道，对每个通道的列进行移动
    for i in range(nC):
        # 对每个通道的图像数据进行列向的裁剪，移动确定的步长
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]

    # 返回裁剪后的张量，仅包含需要的列数
    return inputs[:, :, :, :out_col]

class Illumination_Estimator(nn.Module):
    """
    一个用于估计图像中的照明条件的神经网络模型。
    
    参数:
    - n_fea_middle: 中间特征层的特征数量。
    - n_fea_in: 输入特征的数量，默认为4。
    - n_fea_out: 输出特征的数量，默认为3。
    """

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        """
        初始化Illumination_Estimator网络结构。
        """
        super(Illumination_Estimator, self).__init__()

        # 第一个卷积层，用于将输入特征映射到中间特征空间。
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        # 深度可分离卷积层，用于在中间特征空间内部进行空间特征提取。
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        # 第二个卷积层，用于将中间特征映射到输出特征。
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

        # print("Illumination_Estimator的三个卷积模块已经建立完成！")

        # time.sleep(2)

    def forward(self, img):
        """
        前向传播函数定义。
        
        参数:
        - img: 输入图像，形状为 (b, c=3, h, w)，其中 b 是批量大小, c 是颜色通道数, h 和 w 是图像的高度和宽度。
        
        返回:
        - illu_fea: 照明特征图。
        - illu_map: 输出的照明映射，形状为 (b, c=3, h, w)。
        """
        
        # 计算输入图像每个像素点在所有颜色通道上的平均值。
        mean_c = img.mean(dim=1).unsqueeze(1)  # 形状为 (b, 1, h, w) 对应公式中的Lp，也就是照明先验prior
        # print(f"照明先验的图片大小：{mean_c.shape}")

        # 将原始图像和其平均通道合并作为网络输入。
        input = torch.cat([img, mean_c], dim=1)#对应
        # print("原始图像和其平均通道合并图片大小:",input.shape)

        # 通过第一个卷积层处理。
        x_1 = self.conv1(input)

        # 应用深度可分离卷积提取特征。
        illu_fea = self.depth_conv(x_1)
        # print("照明特征图大小:",illu_fea.shape)

        # 通过第二个卷积层得到最终的照明映射。
        illu_map = self.conv2(illu_fea)
        # print("照明图片大小:",illu_map.shape)
    
        return illu_fea, illu_map

class FeedForward(nn.Module):
    """
    实现一个基于卷积的前馈网络模块，通常用于视觉Transformer结构中。
    这个模块使用1x1卷积扩展特征维度，然后通过3x3卷积在这个扩展的维度上进行处理，最后使用1x1卷积将特征维度降回原来的大小。

    参数:
        dim (int): 输入和输出特征的维度。
        mult (int): 特征维度扩展的倍数，默认为4。
    """
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),  # 使用1x1卷积提升特征维度
            GELU(),  # 使用GELU激活函数增加非线性
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),  # 分组卷积处理，维持特征维度不变，增加特征的局部相关性
            GELU(),  # 再次使用GELU激活函数增加非线性
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),  # 使用1x1卷积降低特征维度回到原始大小
        )

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
        x (tensor): 输入特征，形状为 [b, h, w, c]，其中b是批次大小，h和w是空间维度，c是通道数。

        返回:
        out (tensor): 输出特征，形状与输入相同。
        """
        # 由于PyTorch的卷积期望的输入形状为[b, c, h, w]，需要将通道数从最后一个维度移到第二个维度
        out = self.net(x.permute(0, 3, 1, 2).contiguous())  # 调整输入张量的维度
        return out.permute(0, 2, 3, 1)  # 将输出张量的维度调整回[b, h, w, c]格式

class IGAB(nn.Module):
    """
    实现一个交错组注意力块（Interleaved Group Attention Block，IGAB），该块包含多个注意力和前馈网络层。
    每个块循环地执行自注意力和前馈网络操作。

    参数:
        dim (int): 特征维度，对应于每个输入/输出通道的数量。
        dim_head (int): 每个注意力头的维度。
        heads (int): 注意力机制的头数。
        num_blocks (int): 该层中重复模块的数量。
    """
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2,d_state = 16):

        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IFA(dim_2=dim,dim = dim,num_heads = heads,ffn_expansion_factor=2.66 ,bias=True, LayerNorm_type='WithBias'),
                SS2D(d_model=dim,dropout=0,d_state =d_state),   # 加入Mamba的SS2D模块，输入参数都为源代码指定参数
                PreNorm(dim, FeedForward(dim=dim))  # 预归一化层，后跟前馈网络,这一部分相当于LN+FFN
            ]))

    def forward(self, x, illu_fea):
        """
        前向传播函数处理输入特征和光照特征。

        参数:
            x (Tensor): 输入特征张量，形状为 [b, c, h, w]，其中 b 是批次大小，c 是通道数，h 和 w 是高度和宽度。
            illu_fea (Tensor): 光照特征张量，形状为 [b, c, h, w]。

        返回:
            Tensor: 输出特征张量，形状为 [b, c, h, w]。
        """
        # x = x.permute(0, 2, 3, 1)  # 调整张量维度以匹配预期的输入格式[b, h, w, c]
      
        for (trans,ss2d,ff) in self.blocks:
            y=trans(x,illu_fea).permute(0, 2, 3, 1)
            #应用SS2D模块并进行残差连接
            x=ss2d(y)+ x.permute(0, 2, 3, 1)  #当我创建了一个类之后,如果继承nn并且自己定义了一个forward,那么nn会把hook挂到对象中,直接用对象(forward的参数)就能调用forward函数
            # print("经过ss2d的特征大小",x.shape)
            # 应用前馈网络并进行残差连接
            x = ff(x) + x# bhwc
            x = x.permute(0, 3, 1, 2) #bchw
            # print("模块输出的特征大小",x.shape)
        #print("\n")
        return x


class Denoiser(nn.Module):
    """
    Denoiser 类是一个用于图像去噪的深度神经网络模型。该模型利用编码器-解码器架构进行特征提取和重建，同时结合了
    自注意力机制来增强模型对图像中不同区域的关注能力。

    参数:
    in_dim (int): 输入图像的通道数，默认为3（彩色图像）。
    out_dim (int): 输出图像的通道数，通常与输入通道数相同。
    dim (int): 第一层卷积后的特征维度。
    level (int): 编码器和解码器中的层数。将level设置成2的原因是在论文结构中，维数的变化由C变到2C，再变到4C，总共进行两级的下采样，所以level为2
    num_blocks (list): 每一层中IGAB模块的数量。
    """
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4],d_state= 16):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # 输入投影层，使用卷积层将输入图像的通道数转换为更高的维度以进行后续处理。
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)#第一个卷积，得到的特征向量是F0

        # 编码器，逐层增加特征维度的同时降低空间分辨率，用于捕获更高层次的抽象特征。
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim

       
        for i in range(level):#每一级由于对应的特征维数不同，所以IGAB中的块数也不同，也要采用不同的头数
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim, d_state=d_state),#确定每一次调用IGAB时所需要的块数
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),#由于特征维数加深，所以经过卷积要减少H和W的维数，相当于进行下采样,对应前馈中的FeaDownSample
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)#由于x的特征维数加深，导致分辨率下降，而x后面要进MSA，所以为了保持一直，fea也要进行对应的采样。对应前馈中的IlluFeaDownsample
            ]))
            dim_level *= 2#让其特征维数dim不断加深
            d_state *= 2
        
        # print(f"目前已经建立好的编码器：{self.encoder_layers}")

        # time.sleep(3)

        # neck，处理最深层的特征，使用IGAB模块进行自注意力计算。

        
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1], d_state=d_state)#这一个IGAB对应4C特征维度的

        # print(f"4C特征维度IGAB的建立：{self.bottleneck}，其对应的block为：{num_blocks[-1]}")

        # print(f"目前最大的特征维数：{dim_level}")

        # time.sleep(3)

        # 解码器，逐层减少特征维度的同时增加空间分辨率，用于重建图像。
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),#反卷积，进行上采样，分辨率变大，特征维度变小，对应前馈中的FeaUpSample
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),#与之前编码时候的IGAB输出concatenate后所经过的卷积层
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim, heads=(dim_level // 2) // dim, d_state=d_state)
            ]))
            dim_level //= 2
            d_state //= 2

        # 输出投影层，将解码器的输出特征转换回原始图像的通道数。
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # 使用LeakyReLU激活函数提供非线性处理。
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # 对所有的可训练参数进行权重初始化。
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 初始化权重，为线性层使用截断正态分布，为层归一化设置常数。
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        前向传播函数定义了如何处理输入x及光照特征illu_fea。

        参数:
        x (torch.Tensor): 输入特征张量，形状为 [b, 3, h, w]。
        illu_fea (torch.Tensor): 光照特征张量，大小为(H,W,C)。

        返回:
        torch.Tensor: 去噪后的输出特征张量，形状与x相同。
        """
        # 特征嵌入
        fea = self.embedding(x)#对应图中的F0特征向量

        # print(f"F0的大小：{fea.shape},此时对应illu_fea的大小为：{illu_fea.shape}")#这个大小应该与illu_fea的相同

        count = 0

        # time.sleep(3)

        # print("------------------------------------------开始编码---------------------------------------------")

        # 编码过程
        fea_encoder = []#为了存入每次编码时产生的特征向量，为了后续解码时与解码的特征向量进行concatenate
        illu_fea_list = []#为了存入每次编码时产生的特征向量，为后续解码时候直接调用对应维度的fea使用
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            count+=1
            # print(f"第{count}次进行编码！")
            fea = IGAB(fea, illu_fea)#调用IGAB的前馈网络,此时fea的大小不变
            illu_fea_list.append(illu_fea)#将当前的illu_fea添加到列表中
            fea_encoder.append(fea)#将经过IGAB的特征向量添加到列表中，此时大小还是不变
            fea = FeaDownSample(fea)#进行下采样，此时特征向量分辨率变为原来的一般，特征维度变为原来的两倍
            # print(f"F{count}的大小为：{fea.shape}")
            illu_fea = IlluFeaDownsample(illu_fea)#调整illu_fea的大小，为了使其下次输入的时候能够和特征向量大小对齐

        # 处理最深层的特征
        fea = self.bottleneck(fea, illu_fea)

        # print(f"F'2的大小：{fea.shape}")

        #print("------------------------------------------编码完成---------------------------------------------")

        #time.sleep(3)
        
        #print("------------------------------------------开始解码---------------------------------------------")
        
        count = 0

        # 解码过程
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            count+=1
            #print(f"第{count}次进行解码！")
            fea = FeaUpSample(fea)#进行反卷积，使特征向量分辨率变大，特征维度下降为原来的一半，此时illu_fea还没有变
            #print("")
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))#在这里直接将编码时对应的解码向量concatenate，然后送入卷积层
            illu_fea = illu_fea_list[self.level - 1 - i]#取出这一次所需要的illu_fea
            fea = LeWinBlcok(fea, illu_fea)#送入IGAB模块
            #print(f"F'{2-count}的大小：{fea.shape}")
        
        #print("------------------------------------------解码完成---------------------------------------------")

        # time.sleep(3)

        # 映射到输出维度，并添加原始输入以实现残差连接
        out = self.mapping(fea) + x

        #print(f"最终输出图片Ien的大小为：{out.shape}")    

        return out


class RetinexMamba_Single_Stage(nn.Module):
    """
    定义 Retinex 网络的单一阶段模型，该模型将照明估计和去噪/图像增强任务结合在一个统一的框架中。
    """
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1],d_state = 16):
        """
        初始化 RetinexMamba 单阶段模型。
        
        参数:
            in_channels (int): 输入通道数，通常是3（对于RGB图像）。
            out_channels (int): 输出通道数，通常是3（对于RGB图像）。
            n_feat (int): 特征数，表示网络中间层的特征深度。
            level (int): 网络的深度或级别，影响网络的复杂度和性能。
            num_blocks (list): 每个级别的块数量，用于定义每个级别的重复模块数。
        """
        super(RetinexMamba_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)  # 照明估计器，估计图像的照明成分。
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level, num_blocks=num_blocks, d_state=d_state)  # 编解码获取最终结果
    
    def forward(self, img):
        """
        定义模型的前向传播路径。

        参数:
            img (Tensor): 输入的图像张量，尺寸为 [batch_size, channels, height, width]

        返回:
            output_img (Tensor): 增强后的输出图像张量。
        """
        # 从输入图像中估计照明特征和照明图
        illu_fea, illu_map = self.estimator(img)  # illu_fea: 照明特征; illu_map: 照明图
        
        # 计算输入图像与其照明图的组合，以进行后续增强
        input_img = img * illu_map + img  # 增加照明影响后的图像，为图中的Ilu

        # print(f"Ilu的大小为：{input_img.shape}")
        
        # 使用去噪器进行图像增强，同时利用照明特征
        output_img = self.denoiser(input_img, illu_fea)

        return output_img


class RetinexMamba(nn.Module):
    """
    多阶段 Retinex 图像处理网络，每个阶段都通过 RetinexMamba_Single_Stage 来实现，
    进行图像的照明估计和增强。
    """
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 1, 1], d_state=16):
        """
        初始化 Retinex 图像处理网络。

        参数:
            in_channels (int): 输入图像的通道数，通常为3（RGB图像）。
            out_channels (int): 输出图像的通道数，通常为3（RGB图像）。
            n_feat (int): 特征层数，表示中间特征的深度。
            stage (int): 网络包含的阶段数，每个阶段都使用一个 RetinexMamba_Single_Stage 模块。
            num_blocks (list): 每个阶段的块数，指定每个单阶段中的块数量。
        """
        super(RetinexMamba, self).__init__()
        self.stage = stage  # 网络的阶段数

        # 创建多个 RetinexMamba_Single_Stage 实例，每个实例都作为网络的一个阶段
        modules_body = [RetinexMamba_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks, d_state=d_state)
                        for _ in range(stage)]
        
        # 将所有阶段模块封装成一个顺序模型
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        """
        定义网络的前向传播路径。

        参数:
            x (Tensor): 输入的图像张量，尺寸为 [batch_size, channels, height, width]

        返回:
            out (Tensor): 经过多个阶段处理后的输出图像张量。
        """
        # 通过网络体进行图像处理
        out = self.body(x)

        return out


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    model = RetinexMamba(stage=1,n_feat=40,num_blocks=[1,2,2]).cuda()
    print(model)
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    model(inputs)
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')
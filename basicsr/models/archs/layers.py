import torch.nn as nn

class Mlp(nn.Module):
    """
    在视觉Transformer、MLP-Mixer等网络中使用的多层感知机（MLP）。

    参数:
        in_features (int): 输入特征的数量。
        hidden_features (int, optional): 隐藏层特征的数量。如果未指定，则默认为输入特征的数量。
        out_features (int, optional): 输出特征的数量。如果未指定，则默认为输入特征的数量。
        act_layer (nn.Module, optional): 激活层类型，默认为nn.GELU。
        drop (float, optional): Dropout层的概率，默认为0。
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 如果未指定out_features，则默认为in_features
        hidden_features = hidden_features or in_features  # 如果未指定hidden_features，则默认为in_features

        # 定义第一个全连接层，将输入特征映射到隐藏特征
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活层，通常使用GELU函数
        self.act = act_layer()
        # 第二个全连接层，将隐藏特征映射到输出特征
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout层，用于减少过拟合
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 通过第一个全连接层
        x = self.fc1(x)
        # 应用激活函数
        x = self.act(x)
        # 应用Dropout
        x = self.drop(x)
        # 通过第二个全连接层
        x = self.fc2(x)
        # 再次应用Dropout
        x = self.drop(x)
        return x

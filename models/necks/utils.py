import torch
import torch.nn as nn


class MLPChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPChannelReducer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = in_channels//2

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_channels),
            nn.Sigmoid()  # Sigmoid将输出限制在0到1之间，作为通道的比例
        )

    def forward(self, x):
        # x的形状应该为[B, C, H, W]
        B, C, H, W = x.size()
        assert C == self.in_channels, "输入特征图的通道数与指定的输入通道数不匹配"

        # 将特征图展平为形状[B, C*H*W]
        x = x.view(B, C, -1)
        x = x.mean(dim=2)  # 对每个通道的特征进行平均池化，得到形状[B, C]

        # 使用MLP降低通道数
        channel_ratios = self.mlp(x)  # 形状为[B, out_channels]

        # 根据通道比例重新组合特征图
        reduced_channels = (channel_ratios.unsqueeze(-1) * x.unsqueeze(1)).sum(dim=2)

        return reduced_channels.unsqueeze(-1).unsqueeze(-1)  # 返回形状为[B, out_channels, 1, 1]的特征图
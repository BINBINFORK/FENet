import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    # 通道注意力机制
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),  # 第一层神经元个数为 C/r
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)   # 第二层神经元个数为 C
            )
        self.pool_types = pool_types
        self.conv1 = nn.Conv2d(gate_channels,1,1)

    def forward(self, x):  # 输入F
        channel_att_sum = None
        conv1 = self.conv1(x)
        for pool_type in self.pool_types:
            # if pool_type=='avg':
            #     avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 平均池化为[n,c,1,1]
            #     channel_att_raw = self.mlp( avg_pool )  # 经过独立的两层全连接层
            if pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 最大池化为[n,c,1,1]
                channel_att_raw = self.mlp( max_pool )  # 经过独立的两层全连接层

            if channel_att_sum is None:
               channel_att_sum = channel_att_raw
            # else:
            #     channel_att_sum = channel_att_sum + channel_att_raw  # 将两次全连接层的输出进行相加

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)  # 扩展为[n,c,1,1]，与原矩阵相乘
        return conv1 * scale  # 输出Mc

if __name__ == '__main__':
    x = torch.FloatTensor(3,256,28,28)
    # y = torch.FloatTensor(3,512,16,16)
    b = ChannelGate(256, reduction_ratio=16, pool_types=['max'])
    out = b(x)
    print(out.shape)
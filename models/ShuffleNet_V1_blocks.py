import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleV1Block(nn.Module):
    def __init__(self, inp, oup, *, group, first_group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.group = group

        if stride == 2:
            outputs = oup - inp
        else:
            outputs = oup

        branch_main_1 = [
            # pw   1x1分组卷积降维
            nn.Conv2d(inp, mid_channels, 1, 1, 0, groups=1 if first_group else group, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw   3x3深度卷积
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        ]
        branch_main_2 = [
            # pw-linear   1x1分组卷积升维
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(outputs),
        ]
        self.branch_main_1 = nn.Sequential(*branch_main_1)
        self.branch_main_2 = nn.Sequential(*branch_main_2)

        if stride == 2:  # 下采样，concat拼接
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, old_x):
        '''
        参考shuffleNet模块介绍
        :param old_x:输入特征
        :return:
        '''
        x = old_x
        x_proj = old_x  # copy一份父本进行恒等映射，即短接线连接父本
        x = self.branch_main_1(x)
        if self.group > 1:
            x = self.channel_shuffle(x)
        x = self.branch_main_2(x)
        if self.stride == 1:  # 标准模块直接与父本求和
            return F.relu(x + x_proj)
        elif self.stride == 2:  # 下采样模块需要与父本进行通道concat拼接
            return torch.cat((self.branch_proj(x_proj), F.relu(x)), 1)

    def channel_shuffle(self, x):  # 通道重排
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group
        
        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torchstat import stat

class Block(nn.Module):
    """
     Inverted residual block
    """
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        # 通过 expansion 增大 feature map 的数量
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        # 步长为 1 时，如果 in 和 out 的 feature map 通道不同，用一个卷积改变通道数
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes))
        # 步长为 1 时，如果 in 和 out 的 feature map 通道相同，直接返回输入
        if stride == 1 and in_planes == out_planes:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # 步长为1，加 shortcut 操作
        if self.stride == 1:
            return out + self.shortcut(x)
        # 步长为2，直接输出
        else:
            return out

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    # cfg = [(1, 16, 1, 1),
    #        (6, 24, 2, 1),
    #        (6, 32, 3, 2),
    #        (6, 64, 4, 2),
    #        (6, 96, 3, 1),
    #        (6, 160, 3, 2),
    #        (6, 320, 1, 1)]  # 原MobileNetV2结构

    cfg = [(1,  16, 1, 1),
           (1,  24, 1, 1),
           (1,  32, 1, 2),
           (1,  64, 1, 2),
           (1,  96, 1, 1),
           (1, 160, 1, 2),
           (1, 128, 1, 1)]

    def __init__(self, input_size=32, num_classes=4):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)  # 原文[in_ch=320, out_ch=1280]
        self.bn2 = nn.BatchNorm2d(128)
        self.linear = nn.Linear(128, num_classes)  # in_ch=1280

        # 4. 再经过一次1x1的标准卷积
        self.conv_last = nn.Sequential(
            nn.Conv2d(128, num_classes, 1, 1, 0, bias=False),  # in_ch=1280
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))  # [64, 32, 32, 32]
        out2 = self.layers(out1)  # [64, 320, 4, 4]
        out3 = F.relu(self.bn2(self.conv2(out2)))  # [64, 1280, 4, 4]
        out4 = F.avg_pool2d(out3, 4)  # [64, 1280, 1, 1]
        out5 = self.conv_last(out4)
        out6 = out5.view(out5.size(0), -1)
        # print(out.shape)
        #out7 = self.linear(out6)
        # print(out.shape)
        return out6#,out6, out5, out4, out3, out2


if __name__ == '__main__':
    model = MobileNetV2(num_classes=4)
    x = torch.randn(64, 1, 32, 32)
    y = model(x)
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape, y[4].shape)
    stat(model, (1, 32, 32))

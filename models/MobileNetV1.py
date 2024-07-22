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
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()

        # Depthwise 卷积，3*3 的卷积核，分为 in_planes，即各层单独进行卷积
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)

        # Pointwise 卷积，1*1 的卷积核
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetV1(nn.Module):
    # (128,2) means conv planes=128, stride=2
    # cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1),
    #        (1024, 2), (1024, 1)]  # 原始MobileV1网络结构
    cfg = [(64, 1), (128, 2), (128, 1), (128, 2), (128, 1), (128, 2), (128, 1),
           (128, 2), (128, 1)]
    def __init__(self, input_size=32, num_classes=4):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Sequential(nn.Linear(128, num_classes))
        self.num_class = num_classes
        self.avgpool_test = nn.Sequential(nn.AvgPool2d(kernel_size=1, stride=2))

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x[0]
            stride = x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def dim_show(self, input):
        '''
        测试net函数内每一层的输出维度
        :return:
        '''
        X = input

        print('1.标准卷积first：')
        for layer in self.conv1:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

        print('2.卷积层：')
        for layer in self.layers:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

        print('3.池化层：')
        for layer in self.avgpool_test:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

        print('4.全连接层：')
        X = X.view(X.size(0), -1)
        for layer in self.linear:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

    def forward(self, x):

        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layers(out1)
        out3 = F.avg_pool2d(out2, 2)
        out4 = out3.view(out3.size(0), -1)

        out5 = self.linear(out4)
        #out5_fea = out5.contiguous().view(out5.shape[0], self.num_class, 1, 1)


        return out5#, out5_fea, out3, out2, out1


if __name__ == '__main__':
    models = MobileNetV1(num_classes=4)
    inputs = torch.rand(64, 1, 32, 32)
    outputs = models.dim_show(inputs)
    y = models(x=inputs)
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape, y[4].shape)
    stat(models, (1, 32, 32))

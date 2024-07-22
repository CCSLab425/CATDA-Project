import torch
import torch.nn as nn


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        # 基本分支
        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),

            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        # branch_main = [
        #     # 在PW时先进行下采样
        #     # pw
        #     nn.Conv2d(inp, mid_channels, 1, 2, 0, bias=False),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #
        #     # dw
        #     nn.Conv2d(mid_channels, mid_channels, ksize, 1, pad, groups=mid_channels, bias=False),
        #     nn.BatchNorm2d(mid_channels),
        #
        #     # pw-linear
        #     nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(outputs),
        #     nn.ReLU(inplace=True),
        # ]
        self.branch_main = nn.Sequential(*branch_main)

        # 下采样分支
        if stride == 2:
            branch_proj = [
                # dw， 3x3深度卷积
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),

                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        # Shuffle V2基本模块
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)  # 先进行通道分离，然后进行通道重排
            return torch.cat((x_proj, self.branch_main(x)), 1)  # x_proj走基本模块左分支，x走右分支

        # Shuffle V2下采样模块
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)  # x_proj走下采样模块左分支，x走右分支

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)  # 矩阵转置
        x = x.reshape(2, -1, num_channels // 2, height, width)

        return x[0], x[1]

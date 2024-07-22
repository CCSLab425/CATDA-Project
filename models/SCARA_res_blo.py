import torch
import torch.nn as nn
from torchstat import stat


class InvertedResidual_Block(nn.Module):
    """
    反向残差网络
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_Block, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.stride = stride
        self.identity = stride == 2 and inp == oup
        self.averagePool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        if expand_ratio == 1:
            # 不使用反向残差
            # stride = 1
            self.conv_1 = nn.Sequential(
                # pw
                nn.Conv2d(hidden_dim//2, hidden_dim//2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim//2),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim // 2, oup // 2, 3, stride, 1, groups=oup // 2, bias=False),
                # nn.BatchNorm2d(oup // 2),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(oup // 2, oup // 2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup // 2),
                nn.ReLU6(inplace=True)
            )
            # stride = 2
            self.conv_2 = nn.Sequential(
                # pw
                nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        else:
            # 反向残差，（输入）先升维，（输出）再降维，模块结构为纺锤形
            # stride = 1
            self.conv_1 = nn.Sequential(
                # pw
                nn.Conv2d(inp//2, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.ReLU6(inplace=True),

                # pw-linear
                nn.Conv2d(hidden_dim, oup // 2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup // 2),
                nn.ReLU6(inplace=True),
            )

            # stride = 2
            self.conv_2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

    def dim_show(self, input):
        '''
        测试net函数内每一层的输出维度
        :return:
        '''
        X = input

        print('1.标准卷积first：')

        if self.stride == 1:
            left_branch, right_branch = self.channel_shuffle_1(X)
            for layer in self.conv_1:
                # X = layer(right_branch)
                print(layer.__class__.__name__, 'output shape: \t', X.shape)

        elif self.stride == 2:
            for layer in self.conv_2:
                # print('runing...')
                X = layer(X)
                print(layer.__class__.__name__, 'output shape: \t', X.shape)

    def forward(self, x):

        if self.stride == 1:
            left_branch, right_branch = self.channel_shuffle_1(x)
            right_branch = self.conv_1(right_branch)

            return torch.cat((left_branch, right_branch), 1)

        if self.identity:
            left_branch = self.averagePool(x)
            # left_branch = self.maxPool(x)
            right_branch = self.conv_2(x)
            block_output = torch.cat((left_branch, right_branch), 1)
            block_output = self.channel_shuffle_2(block_output)
            return block_output
        else:
            print('else')
            x = self.conv_2(x)
            return x

    def channel_shuffle_1(self, x):
        """通道重排与通道分离"""
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)  # 矩阵转置
        x = x.reshape(2, -1, num_channels // 2, height, width)

        return x[0], x[1]

    def channel_shuffle_2(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)  # 矩阵转置
        x = x.reshape(2, -1, num_channels // 2, height, width)
        x = torch.cat((x[0], x[1]), 1)
        return x


if __name__ == '__main__':

    inputs = torch.rand(64, 24, 8, 8)
    block = InvertedResidual_Block(inp=24, oup=24, stride=2, expand_ratio=1)
    block.dim_show(inputs)
    block_outputs = block.forward(inputs)
    print('block_outputs.shape = ', block_outputs.shape)
    # stat(model=block, input_size=(24, 32, 32))

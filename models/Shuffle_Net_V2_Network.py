import torch
import torch.nn as nn
from models.Shuffle_Net_V2_Blocks import ShuffleV2Block
import math

from torchstat import stat
from ptflops import get_model_complexity_info


class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=32, n_class=4, model_size='0.5x', in_features=0):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)
        self.stage_repeats = [4, 8, 4]  # 标准ShuffleNetV2每个stage重复次数
        # self.stage_repeats = [2, 4, 2]  # 每个stage重复次数
        # self.stage_repeats = [1, 2, 1]  # 每个stage重复次数
        self.model_size = model_size
        self.in_features = in_features
        self.n_class = n_class

        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]

        # 1. 先经过3x3标准卷积，原始shuffle net V2： stride=2
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True)
        )

        # 2. 最大池化
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # 3. 三个stage
        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            # 每个stage重复调用block单元numrepeat次
            for i in range(numrepeat):
                # 每个stage首先进行stride=2的下采样
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=2))
                # 之后为基本模块
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        # 4. 再经过一次1x1的标准卷积
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.n_class, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.n_class),
            nn.ReLU(inplace=True)
        )

        # 5. 经过全局池化，此处修改nn.AvgPool2d()
        # self.globalpool = nn.Sequential(nn.AvgPool2d(2))  # 原始shuffle net V2
        self.globalpool = nn.Sequential(nn.AvgPool2d(1))
        if self.model_size == '2.0x':
            self.dropout = nn.Dropout(0.2)

        # 6. 最后经过全连接层
        # self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

        if self.in_features != 0:
            self.fc1 = nn.Linear(self.stage_out_channels[-1], self.in_features)
            self.fc2 = nn.Linear(self.in_features, n_class)
        else:
            self.fc = nn.Linear(self.stage_out_channels[-1], n_class)
            # 调整模型各层权重与偏置参数


    def dim_show(self, input):
        '''
        测试net函数内每一层的输出维度
        :return:
        '''
        X = input

        print('1.标准卷积first：')
        for layer in self.first_conv:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

        print('2.最大池化：')
        for layer in self.maxpool:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

        print('3.三个stage：')
        for layer in self.features:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

        print('4.标准卷积last：')
        for layer in self.conv_last:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

        print('5.全局池化：')
        # for layer in self.globalpool:
        #     X = layer(X)
        #     print(layer.__class__.__name__, 'output shape: \t', X.shape)

        print('6.FC：')
        # for layer in self.classifier:
        #     X = layer(X)
        #     print(layer.__class__.__name__, 'output shape: \t', X.shape)

    def forward(self, x, with_ft=False):
        # print('ShuffleNetV2...')
        x1 = self.first_conv(x)
        x2 = self.maxpool(x1)
        x3 = self.features(x2)
        x4 = self.conv_last(x3)

        # x = self.globalpool(x)
        # if self.model_size == '2.0x':
        #     x = self.dropout(x)

        x5 = x4.contiguous().view(-1, self.n_class)

        return x5#, x4, x3, x2, x1

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = ShuffleNetV2(n_class=4)
    # print(model)

    test_data = torch.rand(64, 1, 32, 32)
    model.dim_show(test_data)
    y = model(test_data)
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape, y[4].shape)

    stat(model, (1, 32, 32))
    # flops, params = get_model_complexity_info(model, (1, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)
    # print('Flops:  ', flops)
    # print('Params: ', params)
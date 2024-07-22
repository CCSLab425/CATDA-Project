import torch
import torch.nn as nn
from models.ShuffleNet_V1_blocks import ShuffleV1Block
from torchstat import stat

class ShuffleNetV1(nn.Module):
    def __init__(self, input_size=224, n_class=4, model_size='0.5x', group=None):
        super(ShuffleNetV1, self).__init__()
        print('model size is ', model_size)

        assert group is not None

        # 参考 “不同分组数的shuffleNet网络结构” 进行编写
        self.stage_repeats = [4, 8, 4]  # stage2=1+3，# stage3=1+7，# stage4=1+3，
        # self.stage_repeats = [1, 2, 1]
        self.model_size = model_size
        self.num_class = n_class
        # 不同分组对应的输出通道，可以等比例调整模型尺寸
        if group == 3:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 12, 120, 240, 480]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 240, 480, 960]  # standard
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 360, 720, 1440]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 384, 768, 1536]  # standard
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError

        # 搭建网络层结构，参考 “不同分组数的shuffleNet网络结构” 进行编写

        # 1. Conv1，标准卷积
        input_channel = self.stage_out_channels[1]  # 输入通道数在stage_out_channels[1]中定义
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        # 2. MaxPool
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.features = []

        # 3. 遍历每个stage
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]  # 每个stage重复次数：【4，8，4】
            output_channel = self.stage_out_channels[idxstage+2]

            # 4. 遍历每个block
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                self.features.append(ShuffleV1Block(input_channel, output_channel,
                                            group=group, first_group=first_group,
                                            mid_channels=output_channel // 4, ksize=3, stride=stride))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)  # 将各stage模块进行堆叠

        # 5. 全局池化
        self.globalpool = nn.Sequential(nn.AvgPool2d(1))

        # 6. 分类
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def dim_show(self, input):
        '''
        测试net函数内每一层的输出维度
        :return:
        '''
        X = input

        print('1.标准卷积：')
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

        print('4.全局池化：')
        for layer in self.globalpool:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

        print('5.FC：')
        # for layer in self.classifier:
        #     X = layer(X)
        #     print(layer.__class__.__name__, 'output shape: \t', X.shape)


    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.maxpool(x1)
        x3 = self.features(x2)

        x4 = self.globalpool(x3)
        x5 = x4.contiguous().view(-1, self.stage_out_channels[-1])
        x6 = self.classifier(x5)
        x6_fea = x6.contiguous().view(x6.shape[0], self.num_class, 1, 1)
        return x6#, x6_fea, x4, x3, x2

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
    model = ShuffleNetV1(input_size=32, n_class=4, model_size='0.5x', group=3)
    # print(model)

    test_data = torch.rand(64, 1, 32, 32)
    # test_data = torch.rand(1, 1, 32, 32)
    model.dim_show(test_data)
    y = model(test_data)
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape, y[4].shape)
    stat(model, (1, 32, 32))


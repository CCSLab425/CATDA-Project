import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================================================================= #
#                                  1. MMD loss function
# ================================================================================================================= #
class MMDLoss(nn.Module):
    """
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    """
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        """
        用高斯核函数（RBF）判断两种数据之间的分布差异
        将源域数据和目标域数据转化为核矩阵，即上文中的K
        Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
        Return:
            sum(kernel_val): 多个核矩阵之和
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        """
        线性度量距离
        """
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


if __name__ == '__main__':
    # =============================================================================================================== #
    #                                           2. 嵌入到CNN中代码实现
    # =============================================================================================================== #
    # 先定义一个简单的CNN模型
    class Net(nn.Module):
        """
        计算源域数据和目标域数据的MMD距离
        Params:
        x_in: 输入数据（batch, channel, hight, width）
        Return:
        x_out: 输出数据（batch, n_labes)
        """

        # 这里 x_in：batch=64, channel=3, hight=128, width=128
        #  x_out：batch=64, n_labes=5
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.cls = nn.Conv2d(in_channels=64, out_channels=5, kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.bn1(x)
            x = F.relu(self.conv2(x))
            x = self.bn2(x)
            x = F.relu(self.conv3(x))
            x = self.bn3(x)
            x = F.relu(self.conv4(x))
            x = self.bn4(x)
            x = self.cls(x)
            x = x.view(-1, x.size(1))
            return x
    model = Net()
    source = torch.rand(64, 1, 32, 32)  # 模拟产生batch=64，channel=1， hight=32， width=32 的轴承数据
    target = torch.rand(32, 1, 32, 32)  # 模拟产生batch=32，channel=1， hight=32， width=32 的轴承数据
    source = model(source)
    target = model(target)
    print(source.shape)

    MMD = MMDLoss()
    loss = MMD(source=source, target=target)
    print(loss)

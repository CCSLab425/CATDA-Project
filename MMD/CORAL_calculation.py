"""
功能：计算网络最后1层提取的源域与目标域之间的CORAL距离。
作者: 陈启通
日期：2023年5月23日
"""

import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def CORAL_loss_fn(source_feature, target_feature):
    """
    计算源域和目标域之间的CORAL
    source_feature：源域特征，二维数据
    target_feature：目标域特征，二维数据
    """
    d = source_feature.size(1)
    ns, nt = source_feature.size(0), target_feature.size(0)  # 源域与目标域样本的数量
    source = source_feature.cuda(0)
    target = target_feature.cuda(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(DEVICE) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)  # 源域特征的协方差矩阵

    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)  # 目标域特征的协方差矩阵

    # Frobenius 范数，对应元素的平方和，再开方
    coral_loss = (cs - ct).pow(2).sum().sqrt()
    loss = coral_loss / (4 * d * d)

    return loss

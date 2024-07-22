"""
功能：计算网络最后三层提取的源域与目标域之间的mmd距离。
作者: 陈启通
日期：2023年5月21日
"""
from MMD.MMD_calculation import MMDLoss


def DAN_loss_fn(src_lyr1_fea=None, src_lyr2_fea=None, src_lyr3_fea=None,
                tar_lyr1_fea=None, tar_lyr2_fea=None, tar_lyr3_fea=None):
    """
    计算DAN网络的域适配损失，即最后三层网络输出的源域和目标域特征之间的MK-MMD损失
    src_lyr1_fea: 倒数第1层网络提取的源域特征
    src_lyr2_fea: 倒数第2层网络提取的源域特征
    src_lyr3_fea: 倒数第3层网络提取的源域特征
    tar_lyr1_fea: 倒数第1层网络提取的目标域特征
    tar_lyr2_fea: 倒数第2层网络提取的目标域特征
    tar_lyr3_fea: 倒数第3层网络提取的目标域特征
    """
    mmd_loss = MMDLoss()
    src_layer1_feature = src_lyr1_fea.view(src_lyr1_fea.shape[0], -1)
    src_layer2_feature = src_lyr2_fea.view(src_lyr2_fea.shape[0], -1)
    src_layer3_feature = src_lyr3_fea.view(src_lyr3_fea.shape[0], -1)
    tgt_layer1_feature = tar_lyr1_fea.view(tar_lyr1_fea.shape[0], -1)
    tgt_layer2_feature = tar_lyr2_fea.view(tar_lyr2_fea.shape[0], -1)
    tgt_layer3_feature = tar_lyr3_fea.view(tar_lyr3_fea.shape[0], -1)
    mmd_loss1 = mmd_loss(src_layer1_feature, tgt_layer1_feature)
    mmd_loss2 = mmd_loss(src_layer2_feature, tgt_layer2_feature)
    mmd_loss3 = mmd_loss(src_layer3_feature, tgt_layer3_feature)
    mul_mmd_loss = (mmd_loss1 + mmd_loss2 + mmd_loss3) / 3

    return mul_mmd_loss

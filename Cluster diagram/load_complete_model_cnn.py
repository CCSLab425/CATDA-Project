"""
@author：Qitong Chen
2022.06.28
注意事项：
    加载完整模型进行测试时，需要使用with torch.no_grad():
"""
import os

import torch
# readdata
from os.path import splitext
import scipy.io as io
import scipy
#from files_path import files_path
from models import MobileNetV2
from comparemodels import resnet
from models import Shuffle_Net_V2_Network
from models.SCARA_res_net import SCARA_lightweight_model

def data_reader(datadir, gpu=True):
    """
    read data from mat or other file after readdata.py
    Args:
        datadir: 训练集and测试集的.mat数据地址路径
        gpu: 使用GPU加速

    Returns:训练集和测试集

    """
    print('datadir=====', datadir)
    datatype = splitext(datadir)[1]
    if datatype == '.mat':
        print('datatype is mat')
        data = scipy.io.loadmat(datadir)  # 按照字典方式读取.mat中的数据（训练数据、训练标签、测试数据、测试标签）
        x_train = data['x_train']  # 读取训练数据，数据类型为numpy_array
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
    print('x_train.shape = ', x_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_train.shape = ', y_train.shape)
    print('y_test.shape = ', y_test.shape)

    if datatype == '':
        pass
    # 转化为张量类型
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    return x_train, y_train, x_test, y_test

model_name = 'Lightweight'
sdatadir ="../dataset/bearing_datasets/SUST_500.mat"   # 源域训练集
vdatadir ="../dataset/bearing_datasets/SUST_2000.mat"  # 目标域测试集

xs_train, ys_train, xs_test, ys_test = data_reader(sdatadir)  # 加载源域训练集
xv_train, yv_train, xv_test, yv_test = data_reader(vdatadir)
# 加载训练好的完整模型
load_path = './'
save_path = load_path + 'features/'
source_data_name="500"
target_data_name="2000"
os.makedirs(save_path, exist_ok=True)
model_name="SCARA"
# model = torch.load('{}{}'.format(load_path, model_name))
model = torch.load('../save_dir/compare/05_SUST_500_to_SUST_2000.pth')
# model.cuda(0)  # 使用cuda
# model.eval()
# 开始进行测试
with torch.no_grad():
    # =================================================================================================
    #                                            测试源域
    # =================================================================================================
    # xs_test = xs_train.cuda(0)  # 用xs_train测试
    # ys_test = ys_train.cuda(0)
    xs_test = xs_test.cuda(0)  # 用xs_test测试
    ys_test = ys_test.cuda(0)
    print("///////////////////////////////////////////////////")
    print(xs_test.shape)
    xs_test_pre= model(xs_test)
    print(xs_test_pre.shape)
    xs_test_pre_label = torch.argmax(xs_test_pre, 1)
    ys_test_label = torch.argmax(ys_test, 1)
    xs_acc = (xs_test_pre_label == ys_test_label).sum().float() / len(ys_test_label)
    print('源域xs_test准确率为：', xs_acc)
    # 保存特征
    ys_pre = xs_test_pre.cpu().detach().data.numpy()  # 转化为numpy形式
    ys_pre_dataFile = save_path + 'ys_pre.mat'
    # io.savemat(ys_pre_dataFile, {'ys_pre': ys_pre})
    # 加载特征
    # dataFile = 'E:/01实验室文件/Paper1_Tutorial/实验结果/初始化/ys_pre.mat'
    # data = io.loadmat(dataFile)
    # =======================================================================end

    # =============================================================================================
    #                                          测试目标域
    # =============================================================================================
    idx_start = 0
    idx_end = 2000
    xv_test = xv_test[idx_start:idx_end].cuda(0)
    yv_test = yv_test[idx_start:idx_end].cuda(0)
    xv_test_pre = model(xv_test)
    xv_test_pre_label = torch.argmax(xv_test_pre, 1)
    # print(xv_test_pre_label[0:100])
    yv_test_label = torch.argmax(yv_test, 1)
    val_acc = (xv_test_pre_label == yv_test_label).sum().float() / len(yv_test_label)
    print('目标域xt_test准确率为：', val_acc)
    # 保存特征
    yv_pre = xv_test_pre.cpu().detach().data.numpy()  # 转化为numpy形式
    yv_pre_dataFile = save_path + 'yv_pre.mat'
    # io.savemat(yv_pre_dataFile, {'yv_pre': yv_pre})
    # =======================================================================end


# =================================================================================================
#                                  提取feature_extractor后的特征
# =================================================================================================
activation = {}


def get_activation(name):
    def hook(model, input, output):
        # 如果你想feature的梯度能反向传播，那么去掉 detach（）
        activation[name] = output.detach()
    return hook


#model.sharedNetwork.register_forward_hook(get_activation('feature_extractor'))  # 输出fc2提取的特征
output_s = model(xs_test)
# feature_s = activation['feature_extractor']
# print('feature_s.shape = ', feature_s.shape)

output_v = model(xv_test)
# feature_t = activation['feature_extractor']
# print('feature_t.shape=', feature_t.shape)

# ======================================= -end- ======================================

# 保存源域特征
feature_s = output_s.cpu().detach().data.numpy()  # 转化为numpy形式
feature_t = output_v.cpu().detach().data.numpy()  # 转化为numpy形式

# 保存源域与目标域特征
feature_s_dataFile = save_path + '{}_{}_to_{}_features.mat'.format(
    model_name, source_data_name, target_data_name)
io.savemat(feature_s_dataFile, {'feature_s': feature_s, 'feature_t': feature_t})

# 保存标签
ys_test_label = torch.argmax(ys_test, 1).cpu().detach().data.numpy()  # 转化为numpy形式
yv_test_label = xv_test_pre_label.cpu().detach().data.numpy()  # 转化为numpy形式
# print(ys_test_onehot.shape, yv_test_onehot.shape)
yv_test_dataFile = save_path + 'features_labels.mat'
yv_test_softmax = torch.argmax(yv_test, 1).cpu().detach().data.numpy()
# io.savemat(yv_test_dataFile, {'ys_test_label': ys_test_label, 'yv_test_label': yv_test_label})


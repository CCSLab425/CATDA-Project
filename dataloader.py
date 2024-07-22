# -*- coding: utf-8 -*-
"""
@Author       : 陈启通
@Time         : 2023年5月6日
@Desc         ：20230505 v4.0
@Descriptions ：加载源域与目标域数据
"""

from torch.utils.data import DataLoader, TensorDataset
import torch
from os.path import splitext
import scipy
import scipy.io as io


def data_reader_fn(datadir, gpu=True):
    """
    read data from .mat
    Args:
        datadir: 加载数据文件
    """
    datatype = splitext(datadir)[1]
    if datatype == '.mat':

        data = scipy.io.loadmat(datadir)

        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
    if datatype == '':
        pass

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    y_train = torch.argmax(y_train, 1)  # dim=1 取表示行最大值
    y_test = torch.argmax(y_test, 1)
    return x_train, y_train, x_test, y_test


def dataload(batch_size=64, dataset_path=''):
    """
    加载源域数据集，即参与模型预训练的源域数据集
    :param batch_size: 每次加载样本数
    :param dataset_path: 数据集路径
    :return: loader好的序列数据集
    """
    x_train, y_train, x_test, y_test = data_reader_fn(dataset_path)

    torch_dataset = TensorDataset(x_train, y_train)  # dataset转换成pytorch的格式
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return loader


def test_dataload(batch_size=64, dataset_path=''):
    """
    加载目标域数据集，用于模型测试
    :param batch_size: 每次加载样本数
    :param dataset_path: 数据集路径
    :return: loader好的序列数据集
    """
    xt_train, yt_train, xt_test, yt_test = data_reader_fn(dataset_path)

    torch_dataset = TensorDataset(xt_test, yt_test)  # dataset转换成pytorch的格式
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return loader


def target_dataload(batch_size=64, target_data_name=''):
    t_dataset_path = 'C:/Users/CCSLab/Desktop/lixuan/HC_discrete_datasets_Full_cycle/Current_Feedback/'
    # t_dataset_path = 'C:/Users/CCSLab/Desktop/lixuan/轴承数据集/'

    tdatadir = t_dataset_path + target_data_name
    xt_train, yt_train, xt_test, yt_test = data_reader_fn(tdatadir)

    torch_dataset = TensorDataset(xt_test, yt_test)  # dataset转换成pytorch的格式
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return loader


def source_dataload(batch_size=64, source_data_name=''):
    s_dataset_path = 'D:/DeepLearning/dataset/fd/bearing_datasets/'
    sdatadir = s_dataset_path + source_data_name
    xs_train, ys_train, xs_test, ys_test = data_reader_fn(sdatadir)

    torch_dataset = TensorDataset(xs_test, ys_test)  # dataset转换成pytorch的格式
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    return loader
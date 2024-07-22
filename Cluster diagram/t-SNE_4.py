# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:22:51 2018

@author: Chen Qitong
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from scipy.io import loadmat


def tSNE_(matname):
    """
    对加载的数据特征进行读取与压缩
    Args:
        matname: 数据集名称，一个mat文件中包含源域特征和目标域特征

    Returns:返回压缩后的特征
    """
    # 一个源域，一个目标域
    print('matname = ', matname)
    sfis = loadmat(matname)['feature_s'].reshape(2000, -1)  # 源域, [2000, 96, 2, 2] -> [2000, 384]
    tfit1 = loadmat(matname)['feature_t'].reshape(2000, -1)  # 目标域1, [2000, 96, 2, 2] -> [2000, 384]

    sfis_plus_tfit = np.append(sfis, tfit1, axis=0)  # 将源域与目标域进行合并到一起, size = [4000, 384]
    print('sfis_plus_tfit.shape = ', sfis_plus_tfit.shape)
    tSNE = TSNE(n_components=2, n_iter=400, init='pca', random_state=0)  # 将数据降至二维，训练迭代400次
    tSNE_results = tSNE.fit_transform(sfis_plus_tfit)  # size = [4000, 384]

    return tSNE_results


def plot2D(dic, dataset_name='SBDS', TLname='default'):
    """
    绘制t-SNE图像
    Args:
        dic:经过压缩后的特征
        dataset_name: 需要特征可视化的数据集名称
        TLname:加载的特征文件名称
    """

    tSNEx = dic[:, 0]
    tSNEy = dic[:, 1]
    clist = ['r',  # 红色
             'lightsalmon',  # 日光灯
             'gold',  # 金黄
             'forestgreen',  # 天蓝色 rosybrown
             'royalblue',  # 淡绿色
             'lightseagreen',  # 浅海绿
             'dodgerblue',  # 浅蓝
             'slateblue',  # 板岩蓝
             'violet',  # 紫色
             'purple'  # 紫色的
             ]  # 颜色列表
    clist = ['r', 'lightsalmon', 'lightseagreen', 'slateblue', 'palegreen',
             'lightseagreen', 'lightblue', 'slateblue', 'violet', 'purple']
    label_CWRU = ['Nor', 'Inner', 'Ball', 'Outer']
    label_SBDB = ['Nor', 'Inner', 'Ball', 'Outer']
    label_SDUS = ['Stuck-1', 'Stuck-2', 'Lack', 'Norm']

    if dataset_name == 'CWRU':
        label = label_CWRU
    elif dataset_name == 'SBDB':
        label = label_SBDB
    else:
        label = label_SDUS
    # '.'表示点, ','表示正方形, 'o'圆圈, '钻石, '^'正三角, 'v'倒三角, 'p'五边形, 'D'钻石, '*'五角星, 'h'六边形
    # markerlist = ['.', ',', 'o', 'd', '^', 'v', 'p', 'D', '*', 'h']
    markerlist = ['.', '*', 'o', 'p', 'v', '<', '>', 'D', '*', 'h', 'x', '+']

    num = 602
    sample = 30
    target_idx = 2000

    # 源域和目标域聚类（4类：Normal，Inner，Ball， Outer）
    for i in range(4):
        # 源域，tSNEx = [0 + i*202 : 25 + i*202], tSNEy = [0 + i*202 : 25 + i*202]， 起始索引从0开始
        idx_s_l = 0 + i * num
        idx_s_r = sample + i * num
        # print('i = {}, idx_tSNEx_l = {}, idx_tSNEx_r = {}'.format(i, idx_s_l, idx_s_r))

        plt.scatter(tSNEx[0 + i * num: sample + i * num],
                    tSNEy[0 + i * num: sample + i * num],
                    s=100,  # 表示点大小
                    marker=markerlist[i],  # 散点形状
                    c=clist[i],  # 表示是否填充，c='none'表示无填充
                    edgecolor=clist[i],  # 设置边框颜色
                    label='S-' + label[i],  # 添加图例标签
                    alpha=0.4,  # 透明度
                    )
        # 目标域，tSNEx = [2000 + i*202 : 25 + i*202], tSNEy = [2000 + i*202 : 25 + i*202]， 起始索引从2000开始
        idx_t_l = target_idx + i * num
        idx_t_r = target_idx + sample + i * num
        # print('i = {}, idx_t_l = {}, idx_t_r = {}'.format(i, idx_t_l, idx_t_r))

        plt.scatter(tSNEx[target_idx + i * num: target_idx + sample + i * num],
                    tSNEy[target_idx + i * num: target_idx + sample + i * num],
                    s=100, marker=markerlist[i], c='none', edgecolor=clist[i], label='T-' + label[i], alpha=1,)
    plt.xticks([])
    plt.yticks([])
    plt.title(TLname, family='Times New Roman')


if __name__ == '__main__':
    subgraph_names = [

        'ResNet18',
        'ShuffleNet',
        'MobileNet',
        'LRFFUAL',
        'MLFADA',
        'DDC',
        'CORAL',
        'DAN',
        'CATDA',
                      # 'T0-3',
                      # 'T0-6',
                      # 'T0-9',
                      # 'T3-6',
                      # 'T3-9',
                      # 'T6-9'
                    ]
    features_path = './features/'  # 特征路径
    save_path = './'  # 保存可视化文件路径'../save_dir/process/features_tSNE/'
    #features_path = 'C:/Users/33169/Desktop/跨机器小样本实验/T0-0_festures/'  # 特征路径
    save_path = features_path  # 保存可视化文件路径

    model_name = 'SCARA_lightweight'
    task_name = 'transfer_task'
    save_name = '{}_{}_tSNE'.format(model_name, task_name)
    filelist = os.listdir(features_path)
    j = 0
    plt.figure(figsize=(12, 7))  # 第一个元素为宽，第二个为高，当图像被压缩变形时，可以通过调整画布比例来改正
    a, b = 3, 3  # 子图的数量，a为行，b为列
    for features_file in filelist:
        if features_file[-3:] != 'mat':
            continue
        j += 1
        plt.subplot(a, b, j)
        print('j = ', j)
        tSNEresult = tSNE_(features_path + features_file)
        print('tSNEresult = ', tSNEresult.shape)
        plot2D(tSNEresult, dataset_name='SCARA', TLname=features_file[:-5])
        if j == a*b:
            break
    # ===================================================================================================
    #                                             添加并调整图例
    # ===================================================================================================
    num1 = 1  # 越大越向右移动（若使用TNor, 则设定为1.05）
    num2 = 0  # 向上移动
    num3 = 0  # 1表示在图左侧显示，2表示在图下方显示，3表示在图中底部显示
    num4 = 1  # 表示图例距离图的位置，越大则距离
    plt.legend(bbox_to_anchor=(num1, num2),  # 指定图例在轴的位置
               loc=num3,
               # loc='lower center',
               borderaxespad=num4,  # 轴与图例边框之间的距离
               ncol=10,  # 设置每行的列数，默认按行排列
               prop={"size": 12, 'family': 'Times New Roman'},  # 调整图例字体大小、设置字体样式
               frameon=True,  # 是否保留图例边框
               markerfirst=False,  # 图例与句柄左右相对位置
               # borderpad=0.5,  # 图例边框的内边距
               labelspacing=0.8,  # 图例条目之间的行间距
               columnspacing=0.8,  # 列间距
               handletextpad=0.2,  # 图例句柄和文本之间的间距
               )
    # ===================================================================================================

    plt.savefig(save_path + save_name + '.svg', format='svg')  # 保存图像
    plt.savefig(save_path + save_name + '.jpg', dpi=600)  # 保存图像
    plt.show()




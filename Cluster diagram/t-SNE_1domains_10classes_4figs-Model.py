# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:22:51 2018

@author: 李奇
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
    sfis = loadmat(matname)['feature_s']#.reshape(2000, -1)  # 源域, [2000, 96, 2, 2] -> [2000, 384]
    print(sfis.shape)
    tfit1 = loadmat(matname)['feature_t'].reshape(2000, -1)  # 目标域1, [2000, 96, 2, 2] -> [2000, 384]
    sfis_plus_tfit = np.append(sfis, tfit1, axis=0)  # 将源域与目标域进行合并到一起, size = [4000, 384]
    # print('sfis_plus_tfit.shape = ', sfis_plus_tfit.shape)
    #print(loadmat(matname))
    #sfis_plus_tfit = loadmat(matname)['feature_s']  # 将源域与目标域进行合并到一起, size = [4000, 384]
    # print(sfis_plus_tfit)
    ts = TSNE(n_components=2, n_iter=1000, init='pca', random_state=0, learning_rate=100)
    # tSNE_results = tSNE.fit_transform(sfis_plus_tfit)  # size = [4000, 384]
    tSNE_results = ts.fit_transform(sfis_plus_tfit)

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
    label_CWRU = ['Nor', 'I07', 'I14', 'I21', 'B07', 'B14', 'B21', 'O07', 'O14', 'O21']
    label_SBDB = ['Nor', 'I02', 'I04', 'I06', 'B02', 'B04', 'B06', 'O02', 'O04', 'O06']
    label_SDUS = ['Nor', 'I02', 'I04', 'I06', 'B02', 'B04', 'B06', 'O02', 'O04', 'O06']
    # label_SDUS = ['Nor', 'Nor', 'I02', 'I02', 'I04', 'I04', 'I06', 'I06', 'B02', 'B02',
    #               'B04', 'B04', 'B06', 'B06', 'O02', 'O02', 'S_O04', 'O06', 'O04', 'O06']
    if dataset_name == 'CWRU':
        label = label_CWRU
    elif dataset_name == 'SBDB':
        label = label_SBDB
    else:
        label = label_SDUS
    # '.'表示点, ','表示正方形, 'o'圆圈, '钻石, '^'正三角, 'v'倒三角, 'p'五边形, 'D'钻石, '*'五角星, 'h'六边形
    markerlist = ['.', ',', 'o', 'd', '^', 'v', 'p', 'D', '*', 'h']

    num = 202
    # num = 602
    sample = 25
    target_idx = 2000 #2000
    # 源域和目标域聚类（10类：Normal，Inner*3，Ball*3， Outer*3）
    for i in range(10):
        plt.scatter(tSNEx[i * num:  sample + i * num],
                    tSNEy[i * num: sample + i * num],
                    s=100, marker=markerlist[i], c='None', edgecolor=clist[i], label=label[i], alpha=1,)

        plt.scatter(tSNEx[target_idx + i * num: target_idx + sample + i * num],
                    tSNEy[target_idx + i * num: target_idx + sample + i * num],
                    s=100, marker=markerlist[i], c=clist[i], edgecolor=clist[i], label=label[i], alpha=0.4,)
    plt.xticks([])
    plt.yticks([])
    plt.title(TLname, family='Times New Roman', size=12)


if __name__ == '__main__':
    model_name = [
                # 'ResNet18',
                # 'ShuffleNet',
                # 'MobileNet',
                'CATDA without YHMMD',
                'CATDA without MMD2',
                'CATDA without MMD1',
                # 'LRFFUAL',
                # 'MLFADA',
                # 'DDC',
                # 'CORAL',
                # 'DAN',
                'CATDA',
                  # 'T500-1000',
                  # 'T500-1500',
                  # 'T500-2000',
                  # 'T1000-1500',
                  # 'T1000-2000',
                  # 'T1500-2000'
                  ]

    features_path = './features/消融聚类/'  # 特征路径
    save_path = './'  # 保存可视化文件路径

    filelist = os.listdir(features_path)
    j = 0
    plt.figure(figsize=(10, 8))  # 当图像被压缩变形时，可以通过调整画布比例来改正
    for features_file in filelist:
        j += 1
        if j == 1:
            plt.subplot(221)
        elif j == 2:
            plt.subplot(222)
        elif j == 3:
            plt.subplot(223)
        elif j == 4:
            plt.subplot(224)
        # elif j == 5:
        #     plt.subplot(335)
        # elif j == 6:
        #     plt.subplot(336)
        # elif j == 7:
        #     plt.subplot(337)
        # elif j == 8:
        #     plt.subplot(338)
        # elif j == 9:
        #     plt.subplot(339)
        print(features_file)
        tSNEresult = tSNE_(features_path + features_file)
        print('tSNEresult = ', tSNEresult.shape)
        # plot2D(tSNEresult, dataset_name='SDUS', TLname=features_file[:-4])
        plot2D(tSNEresult, dataset_name='SDUS', TLname=model_name[j-1])
        plt.subplots_adjust(top=0.88, bottom=0.235, left=0.125, right=0.9, hspace=0.2, wspace=0.2)

    # ===================================================================================================
    #                                             添加并调整图例
    # ===================================================================================================
    num1 = 1.07  # 越大越向右移动（若使用T_Nor, 则设定为1.3）
    num2 = 0  # 向上移动
    num3 = 0  # 1表示在图左侧显示，2表示在图下方显示，3表示在图中底部显示
    num4 = 1  # 表示图例距离图的位置，越大则距离
    plt.legend(bbox_to_anchor=(num1, num2),   # 指定图例在轴的位置
               loc=num3,
               # loc='lower center',
               borderaxespad=num4,  # 轴与图例边框之间的距离
               ncol=10,  # 设置每行的列数，默认按行排列
               prop={"size": 12, 'family': 'Times New Roman'},  # 调整图例字体大小、设置字体样式
               frameon=True,  # 是否保留图例边框
               markerfirst=False,  # 图例与句柄左右相对位置
               # borderpad=0.5,  # 图例边框的内边距
               labelspacing=0.8,  # 图例条目之间的行间距
               columnspacing=0.9,  # 列间距
               handletextpad=0.2,  # 图例句柄和文本之间的间距
               )
    # ===================================================================================================

    plt.savefig(save_path + 'robustness_fig' + '.svg', format='svg')  # 保存图像
    plt.savefig('./' + 'robustness_fig' + '.jpg', dpi=600)  # bbox_inches='tight'可确保标签信息显示全
    plt.show()




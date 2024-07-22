"""
@author：Qitong Chen
2022.07.03
功能：根据预测标签与真实标签画混淆矩阵
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from d2l import torch as d2l
from matplotlib import ticker, cm


class DrawConfusionMatrix:
    def __init__(self, labels_name, normalize=True, save_path='./', save_name='fig.svg', best_acc=0, j=0, load='', signal_abbr=''):
        """
        初始化
        Args:
            labels_name: 标签名称
            normalize: 是否设元素为百分比形式
        """
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")
        self.matrix_new = np.zeros((self.num_classes, self.num_classes), dtype="float32")
        self.save_path = save_path
        self.save_name = save_name
        self.total_acc = best_acc
        self.font_size = 6.5
        self.sub_fig = j
        self.load = load

    def update(self, predicts, labels):
        """
            混淆矩阵是将所有数据的label和predict整理而画的，但实际中往往是分成多个iter来推测batch_size个数据，
            所以需要update()函数来将每一次的label和predict值保存进去，模型推理完成后，再调用draw()函数画出混淆矩阵并保存为图片
        Args:
            predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
            labels:  一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)

        Returns:

        """
        for predict, label in zip(predicts, labels):
            # print('predict = ', predict)
            self.matrix[predict, label] += 1
            # print(self.matrix)

    def getMatrix(self, normalize=True):
        """
        根据传入的normalize判断要进行percent的转换，
        如果normalize为True，则矩阵元素转换为百分比形式，
        如果normalize为False，则矩阵元素就为数量
        Returns:返回一个以百分比或者数量为元素的矩阵

        """
        # matrix_0 = np.array([0., 0., 0., 0.])
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算,200
            print('per_sum = ', per_sum)

            for i in range(self.num_classes):
                self.matrix[i] = (self.matrix[i] / per_sum[i])  # 百分比转换

            self.matrix = np.around(self.matrix, 4)   # 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0

        return self.matrix

    def drawMatrix(self):
        """
        颜色条参考:
            cmap='https://www.cnblogs.com/ceason/articles/14177807.html'
        Returns:
        """
        plt.subplot(3, 2, j)
        matrix = self.getMatrix(self.normalize)
        print('self.matrix = \n', matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues, vmin=0, vmax=1)  # 仅画出颜色格子，没有值, Blues, Pastel2, GnBu
        # if self.sub_fig == 10 or self.sub_fig == 11 or self.sub_fig == 12:
        #     plt.xlabel("Predict label", size=self.font_size, family='Times New Roman')  # x轴标签
        # if self.sub_fig == 1 or self.sub_fig == 4 or self.sub_fig == 7 or self.sub_fig == 10:
        #     plt.ylabel("Truth label", size=self.font_size, family='Times New Roman')  # y轴标签
        plt.xlabel("Predicted labels", size=10, family='Times New Roman')  # x轴标签
        plt.ylabel("True labels", size=10, family='Times New Roman')  # y轴标签
        plt.yticks(range(self.num_classes), self.labels_name[0:10], family='Times New Roman', fontsize=self.font_size)  # y轴坐标值
        plt.xticks(range(self.num_classes), self.labels_name[0:10], rotation=0, family='Times New Roman', fontsize=self.font_size)  # x轴坐标值
        accuracy = 0
        for x in range(self.num_classes):
            for y in range(self.num_classes):

                value = float(format('%.5f' % matrix[y][x]))  # 数值处理
                # print('[{}, {}], value = {}'.format(x, y, value))
                if x == y:
                    accuracy += value
                if value == 0:  # 如果值等于0，则省去小数点后的数值
                    value = 0
                else:
                    value = float(format('%.5f' % matrix[y][x]))  # 数值处理

                if value > 0.6:  # 如果值大于0.6，用白色字体显示
                    plt.text(x, y, value, verticalalignment='center', horizontalalignment='center',
                             color='white', family='Times New Roman', fontsize=self.font_size)  # 写值
                else:
                    plt.text(x, y, value, verticalalignment='center', horizontalalignment='center',
                             family='Times New Roman', fontsize=self.font_size)  # 写值
        total_accuracy = accuracy / 10
        total_accuracy = round(total_accuracy, 4)
        print('total_accuracy = ', total_accuracy)
        if self.total_acc == 0:  # 直接加载csv文件进行计算实际的准确率
            # task = 'T' + self.save_name[6] + ' - ' + self.save_name[-5]
            total_acc = total_accuracy * 100
            total_acc = str(total_acc)
            if len(total_acc) > 6:
                total_acc = total_acc[0:6]
            # title = task + ' (Total acc = ' + total_acc + '%)'
            title = '{} (Acc = {}%)'.format(self.load, total_acc)
            plt.title(title, size=self.font_size, family='Times New Roman')  # title
        else:  # 通过训练程序传入准确率
            task = 'T' + self.save_name[6] + ' - ' + self.save_name[-5]
            total_acc = self.total_acc * 100
            total_acc = str(total_acc)
            if len(total_acc) > 6:
                total_acc = total_acc[0:6]
            title = task + ' (Total acc = ' + total_acc + '%)'
            plt.title(title, size=self.font_size, family='Times New Roman')  # title

        plt.rcParams['font.family'] = 'Times New Roman'  # 设置色条的字体
        plt.rcParams['font.size'] = self.font_size  # 设置色条的字体大小
        # plt.colorbar()  # 色条
        if self.sub_fig == 6:
            plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
            plt.savefig(self.save_path + self.save_name + '.jpg', bbox_inches='tight',
                        dpi=300)  # bbox_inches='tight'可确保标签信息显示全
            plt.savefig(self.save_path + self.save_name + '.svg', format='svg')  # 保存图像
            plt.show()


def read_csv_data(csv_path):
    """
    读取csv文件内容
    Args:
        csv_path: csv文件路径

    Returns:读取的csv数据

    """
    data = pd.read_csv(csv_path)

    return data


if __name__ == '__main__':
    #loads = ['T500', 'T1000', 'T1500', 'T2000']
    #loads = ['T0-3', 'T0-6', 'T0-9', 'T3-6', 'T3-9', 'T6-9']
    #loads = [(0, 3), (0, 6), (0, 9), (3, 6), (3, 9), (6, 9)]
    loads = [(500, 1000), (500, 1500), (500, 2000), (1000, 1500), (1000, 2000), (1500, 2000)]
    plt.figure(figsize=(5,8))  # 当图像被压缩变形时，可以通过调整画布比例来改正
    j = 0
    for load in loads:
        j += 1

        source_data_name = f"{load[0]}"
        target_data_name = f"{load[1]}"
        csv_name = f'{source_data_name}_to_{target_data_name}'
        csv_name1= f'T{load[0]}-{load[1]}'
        csv_path = './save_dir/'
        save_path = csv_path
        #csv_name = '3kg_to_6kg'
        save_name = 'Variable_load_confusion_matrix'
        data = read_csv_data(csv_path + csv_name + '.csv')  # 读取csv文件中的所有数据

        yt_label = torch.tensor(data.yt_label.values, dtype=d2l.int32)  # 读取csv文件yt_lable列的数据，size = [2000]
        yt_label_pre = torch.tensor(data.yt_pre_label.values, dtype=d2l.int32)
        print(yt_label.shape)
        print(yt_label_pre.shape)
        #labels_name = ['Stuck1', 'Stuck2', 'Lack', 'Norm']
        labels_name = ['Nor', 'IR1', 'IR2', 'IR3', 'BF1', 'BF2', 'BF3', 'OR1', 'OR2', 'OR3']

        drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name, save_path=save_path, save_name=save_name,
                                                  j=j, load=csv_name1)  # 实例化
        drawconfusionmatrix.update(yt_label, yt_label_pre)  # 将新批次的predict和label更新（保存）
        drawconfusionmatrix.drawMatrix()  # 根据所有predict和label，画出混淆矩阵

        confusion_mat = drawconfusionmatrix.getMatrix()  # 也可以使用该函数获取混淆矩阵(ndarray)
        # print('confusion_mat = ', confusion_mat)


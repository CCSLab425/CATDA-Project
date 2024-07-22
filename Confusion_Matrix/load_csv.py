"""
2022.07.03
功能：读取csv文件中的数据
"""
import pandas as pd
import torch
from d2l import torch as d2l


def read_csv_data(csv_path):
    """
    读取csv文件内容
    Args:
        csv_path: csv文件路径

    Returns:读取的csv数据

    """
    data = pd.read_csv(csv_path)

    return data


csv_path = 'E:/01实验室文件/Paper1_Tutorial/Domain-Specific Batch Normalization for Unsupervised Domain Adaptation/' \
          '轻量化模型备份/01详细实验结果/混淆矩阵/' \
           'SBDS_0K_10_To_SBDS_2K_10_prediction_lable.csv'

data = read_csv_data(csv_path)  # 读取csv文件中的所有数据

# print(data.iloc[0:, [0, 1, 2]])  # # 从第0行开始，读取csv文件第0，1，2列数据
# yt_lable = torch.tensor(data.yt_lable.values, dtype=d2l.int32)  # 读取csv文件yt_lable列的数据，size = [2000]
yt_lable = torch.tensor(data.iloc[0:, [1]].values, dtype=d2l.int32)  # 从第0行开始，读取csv文件第一列数据的值，size = [2000, 1]
yt_pre_lable = torch.tensor(data.iloc[0:, [2]].values, dtype=d2l.int32)
print(yt_lable.shape)
print(yt_pre_lable.shape)


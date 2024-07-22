import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
plt.rc('font',family='Times New Roman')
if __name__ == '__main__':
    csvpath = './save_dir/'
    savepath = csvpath
    subgraph_names = ['SCARA_0kg_4_to_SCARA_3kg_4',
                      'SCARA_0kg_4_to_SCARA_6kg_4',
                      'SCARA_0kg_4_to_SCARA_9kg_4',
                      'SCARA_3kg_4_to_SCARA_6kg_4',
                      'SCARA_3kg_4_to_SCARA_9kg_4',
                      'SCARA_6kg_4_to_SCARA_9kg_4']
    # subgraph_names = ['SUST_500_to_SUST_1000',
    #                   'SUST_500_to_SUST_1500',
    #                   'SUST_500_to_SUST_2000',
    #                   'SUST_1000_to_SUST_1500',
    #                   'SUST_1000_to_SUST_2000',
    #                   'SUST_1500_to_SUST_2000']

    j = 0
    index=0
    # fig = plt.figure(figsize=(10,12))
    plt.figure(figsize=(18, 8))
    for subgrath in subgraph_names:
        index+=1
        j += 1
        # ax = plt.subplot(3, 2, j)
        ax = plt.subplot(2, 3, j)
        ax2 = ax.twinx()
        data = pd.read_csv(csvpath + subgrath + '.csv')

        ax.plot(data.Source_test_loss, label='Source_test_loss')
        ax.plot(data.yhmmd_loss, label='YHMMD_loss')
        #ax.plot(data.total_loss, label='total_loss')
        #ax.plot(data.src_and_tgt_mmd_loss, label='MMD_loss')
        #ax.plot(data.cmmd_loss, label='$\lambda_2$ CMMD_loss')

        ax2.plot(data.Source_test_acc, label='Source_test_acc', color='c')
        ax2.plot(data.Target_test_acc, label='Target_test_acc', color='m')

        # axins = ax.inset_axes((0.3, 0.2, 0.5, 0.3))
        # # axins = ax.inset_axes((0.3, 0.2, 0.1, 0.3))
        # axins.plot(data.Source_test_loss, label='Source_test_loss')
        # axins.plot(data.src_and_tgt_mmd_loss, label='MMD_loss')
        #axins.plot(data.cmmd_loss, label='CMMD_loss')
        # axins.set_xlim(180, 200)
        # axins.set_ylim(0, 0.0005)
        # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='none', lw=1)
        # ax.plot([56, 180], [0.06, 0.001], c='k', linestyle='--')
        # ax.plot([165, 200], [0.06, 0.001], c='k', linestyle='--')

        labels = ['Source_loss', 'YHMMD_loss', 'Source_acc', 'Target_acc']

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_ylim(0,5)
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1.1)

        ax.set_title('T'+subgrath[6]+'-'+subgrath[-5])

        # if(index<=3):
        #     ax.set_title('T' + subgrath[5:8]  + '-' + subgrath[-4:])
        # else:
        #     ax.set_title('T' + subgrath[5:9] + '-' + subgrath[-4:])
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    # plt.legend(labels)
    num1 = 1.07  # 越大越向右移动（若使用TNor, 则设定为1.05）
    num2 = -0.15  # 向上移动
    num3 = 0  # 1表示在图左侧显示，2表示在图下方显示，3表示在图中底部显示
    num4 = 1  # 表示图例距离图的位置，越大则距离
    plt.legend(bbox_to_anchor=(0.8, num2),  # 指定图例在轴的位置
              loc=num3,
              # loc='lower center',
              borderaxespad=num4,  # 轴与图例边框之间的距离
              ncol=10,  # 设置每行的列数，默认按行排列
              prop={"size": 12, 'family': 'Times New Roman'},  # 调整图例字体大小、设置字体样式
              frameon=False,  # 是否保留图例边框
              markerfirst=False,  # 图例与句柄左右相对位置
              # borderpad=0.5,  # 图例边框的内边距
              labelspacing=0.8,  # 图例条目之间的行间距
              columnspacing=6,  # 列间距
              handletextpad=1,  # 图例句柄和文本之间的间距
              )
    ax.legend(bbox_to_anchor=(-0.6, num2),  # 指定图例在轴的位置
               loc=num3,
               # loc='lower center',
               borderaxespad=num4,  # 轴与图例边框之间的距离
               ncol=10,  # 设置每行的列数，默认按行排列
               prop={"size": 12, 'family': 'Times New Roman'},  # 调整图例字体大小、设置字体样式
               frameon=False,  # 是否保留图例边框
               markerfirst=False,  # 图例与句柄左右相对位置
               # borderpad=0.5,  # 图例边框的内边距
               labelspacing=0.8,  # 图例条目之间的行间距
               columnspacing=6,  # 列间距
               handletextpad=1,  # 图例句柄和文本之间的间距
               )

    plt.savefig(savepath+'LineShow.pdf')
    plt.show()



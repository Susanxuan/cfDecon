import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # 导入seaborn库


def main():
    for r1 in [[128, 72, 48, 12]]:  # [128, 72, 48, 12], [128, 84, 52, 14]]:  # [128, 72, 48, 12]
        # 加载数据
        input_path = "ALS.txt"
        simi = 1
        mode = "overall"
        file = "0.3False0.4" #"0.3True0.1", "0.3True0.2", "0.3True0.3"
        for alpha, beta in [(20, 1)]:
            #(20, 1), (10, 1), (10, 10), (10, 20), (1, 20), (1, 10), (20, 20)
            alpha_est_path = './results/test' + input_path + str(simi) + mode + file + str(r1) + str(alpha) + str(
                beta) + 'alpha_est.pkl'
            # alpha_est_path = './results/testcfsort' + input_path + str(simi) + mode + 'alpha_est.pkl'
            # alpha_est_path = './results/testcelfeer' + input_path + 'alpha_est.pkl'
            with open(alpha_est_path, "rb") as fin:
                alpha_est = pickle.load(fin)

            # 确认数据形状
            print('alpha_est', alpha_est.shape)

            # 细胞类型
            cell_types = ['dendritic', 'endothelial', 'eosinophil', 'erythroblast', 'macrophage', 'monocyte',
                          'neutrophil',
                          'placenta', 'tcell', 'adipose', 'brain', 'fibroblast', 'heart', 'hepatocyte', 'lung',
                          'mammary',
                          'megakaryocyte', 'skeletal', 'small_intestine']

            # 创建DataFrame
            data_list = []
            for i, cell_type in enumerate(cell_types):
                for j in range(7, -1, -1):  # 反向迭代
                    group = 'ALS' if j < 4 else 'CTRL'
                    data_list.append({'Cell Type': cell_type, 'Group': group, 'Proportion': alpha_est[j, i]})

            data = pd.DataFrame(data_list)

            # 绘制箱图
            plt.figure(figsize=(16, 10))
            ax = sns.boxplot(y='Cell Type', x='Proportion', hue='Group', data=data, showfliers=False)

            # 设置横轴范围
            ax.set_xlim(0, 0.6)

            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
            # 自定义图表
            plt.title('Proportion of Cell Types in ALS and CTRL Groups', fontsize=16, fontweight='bold')  # 加粗标题
            plt.ylabel('Cell types', fontsize=12, fontweight='bold')  # 加粗y轴标签
            plt.xlabel('Proportions', fontsize=12, fontweight='bold')  # 加粗y轴标签
            # 加粗坐标轴
            plt.gca().spines['top'].set_linewidth(2)
            plt.gca().spines['right'].set_linewidth(2)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)

            plt.xticks(fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12, fontweight='bold')

            # 添加图例
            # plt.legend([plt.Rectangle((0, 0), 1, 1, fc='blue', ec='black', linewidth=2),
            #             plt.Rectangle((0, 0), 1, 1, fc='orange', ec='black', linewidth=2)],
            #            ['CTRL', 'ALS'], loc='upper right')

            plt.legend(title='Groups', fontsize=12, title_fontsize=12, loc='upper right')

            plt.tight_layout()
            plt.savefig('./results/figure/test' + input_path + str(simi) + mode + file + str(r1) + '.png')
            # plt.savefig('./results/figure/testcfsort' + input_path + '.png')
            # plt.savefig('./results/figure/testcelfeer' + input_path + '.png')
            plt.show()



if __name__ == "__main__":
    main()

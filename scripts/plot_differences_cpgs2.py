import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_combined_heatmaps(coad_avg, normal_avg, cell_type, save_path, r1):
    # Create an empty array for the combined data
    combined_data = np.zeros((1900, 10))

    # Interleave COAD and Normal data by channels
    combined_data[:, 0::2] = coad_avg
    combined_data[:, 1::2] = normal_avg

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(combined_data, cmap='coolwarm', cbar=True)

    heatmap = sns.heatmap(combined_data, cmap='coolwarm', cbar=False, ax=ax)

    # 获取颜色条对象
    cbar = heatmap.collections[0].colorbar

    # 调整颜色条的位置
    cbar.ax.yaxis.set_ticks_position('left')  # 将刻度线放在左侧
    cbar.ax.set_ylabel('', rotation=270, labelpad=15)  # 如果需要标签，可以在这里添加

    # 加粗颜色条上的数字
    cbar.ax.tick_params(axis='y', labelsize=10, width=2, length=6)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')

    # 调整颜色条的位置（使其更靠近热力图）
    plt.tight_layout()
    cbar.ax.set_position([ax.get_position().x1+0.01,
                          ax.get_position().y0,
                          0.02,
                          ax.get_position().height])

    cell_types = ['dendritic', 'endothelial', 'eosinophil', 'erythroblast', 'macrophage', 'monocyte', 'neutrophil',
                  'placenta', 'tcell', 'adipose', 'brain', 'fibroblast', 'heart', 'hepatocyte', 'lung', 'mammary',
                  'megakaryocyte', 'skeletal', 'small_intestine']
    # Add channel labels
    channel_labels = ['COAD_0', 'Normal_0', 'COAD_1', 'Normal_1', 'COAD_2', 'Normal_2', 'COAD_3', 'Normal_3', 'COAD_4', 'Normal_4']
    plt.xticks(ticks=np.arange(0.5, 10.5, 1), labels=channel_labels)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel('Channels (Interleaved COAD and Normal)', fontsize=12, fontweight='bold')
    plt.ylabel('CpGs Positions', fontsize=12, fontweight='bold')
    plt.title(f'Cell Type: {cell_types[cell_type]}, COAD vs Normal Average', fontsize=16, fontweight='bold')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'{save_path}/{r1}_{cell_type}_COAD_vs_Normal_Combined_Average.png')
    plt.close()

def main():
    input_path1 = "COAD_ALS.txt" # 52 individuals
    input_path2 = "Normal_ALS.txt" # 193 individuals
    simi = 1
    mode = "high-resolution"
    for r1 in [[128, 72, 48, 12]]:  # [0], [128, 72, 48, 12], [128, 84, 52, 14]

        beta_est_path1 = './results/test' + input_path1 + str(simi) + mode + str(r1) + 'beta_est.pkl'
        with open(beta_est_path1, "rb") as fin:
            beta_est1 = pickle.load(fin)

        beta_est_path2 = './results/test' + input_path2 + str(simi) + mode + str(r1) + 'beta_est.pkl'
        with open(beta_est_path2, "rb") as fin:
            beta_est2 = pickle.load(fin)

        # 创建保存路径
        save_path = './results/heatmaps'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 遍历所有细胞类型
        for cell_type in beta_est1.keys():
            coad_data = beta_est1[cell_type]
            normal_data = beta_est2[cell_type]

            # 计算平均值
            coad_avg = coad_data.mean(axis=0)
            normal_avg = normal_data.mean(axis=0)

            # 生成热图
            plot_combined_heatmaps(coad_avg, normal_avg, cell_type, save_path, r1)

if __name__ == "__main__":
    main()

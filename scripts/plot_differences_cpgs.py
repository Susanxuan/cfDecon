import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_combined_heatmaps(beta_est, cell_type, save_path, r1, input_path):
    als_data = beta_est[:4, :, :].mean(axis=0)
    ctr_data = beta_est[4:, :, :].mean(axis=0)

    # Create an empty array for the combined data
    combined_data = np.zeros((1900, 10))

    # Interleave ALS and CTR data by channels
    combined_data[:, 0::2] = als_data
    combined_data[:, 1::2] = ctr_data
    cell_types = ['dendritic', 'endothelial', 'eosinophil', 'erythroblast', 'macrophage', 'monocyte', 'neutrophil',
                  'placenta', 'tcell', 'adipose', 'brain', 'fibroblast', 'heart', 'hepatocyte', 'lung', 'mammary',
                  'megakaryocyte', 'skeletal', 'small_intestine']
    plt.figure(figsize=(20, 10))
    sns.heatmap(combined_data, cmap='coolwarm', cbar=True)
    plt.title(f'Cell Type: {cell_types[cell_type]}, ALS vs CTR Average')
    plt.xlabel('Channels (Interleaved ALS and CTR)')
    plt.ylabel('Positions')
    plt.savefig(f'{save_path}/new2{input_path}_{r1}_{cell_type}_Combined_Average.png')
    plt.close()


def main():
    input_path = "ALS.txt"
    simi = 1
    mode = "high-resolution"
    file = "0.3false0.4"
    alpha = 20
    beta = 1
    for r1 in [[128, 72, 48, 12]]:  # [0], [128, 72, 48, 12], [128, 84, 52, 14]

        beta_est_path = './results/newtest' + input_path + str(simi) + mode + file + str(r1) + str(alpha) + str(beta)+ 'beta_est.pkl'
        # beta_est_path = './results/' + 'beta_est.pkl'
        with open(beta_est_path, "rb") as fin:
            beta_est = pickle.load(fin)

        # 创建保存路径
        save_path = './results/heatmaps'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 遍历所有细胞类型
        for cell_type, data in beta_est.items():
            plot_combined_heatmaps(data, cell_type, save_path, r1, input_path)


if __name__ == "__main__":
    main()

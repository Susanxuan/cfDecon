import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd


def get_cpg_indices_and_labels(bed_file_path):
    input_path = "data/ALS_WBC_35.txt"
    data_df = pd.read_csv(input_path, delimiter="\t", header=None, skiprows=1)
    bed_data = pd.read_csv(bed_file_path, sep='\t', header=None, names=['chr', 'start', 'end'])

    data_tuples = list(zip(data_df[0], data_df[1], data_df[2]))
    bed_tuples = list(zip(bed_data['chr'], bed_data['start'], bed_data['end']))

    cpg_indices = []
    cpg_labels = []
    for i, row in enumerate(data_tuples):
        if row in bed_tuples:
            idx = bed_tuples.index(row)
            cpg_indices.append(i)
            cpg_labels.append(f"{bed_data['start'][idx]}-{bed_data['end'][idx]}")  # 获取 start-end

    return cpg_indices, cpg_labels


def plot_combined_heatmaps(coad_avg, normal_avg, cell_type, save_path, r1, fix, input_path1, input_path2, bed_file_path):

    cpg_indices, cpg_labels = get_cpg_indices_and_labels(bed_file_path)

    coad_avg_filtered = coad_avg[cpg_indices]
    normal_avg_filtered = normal_avg[cpg_indices]

    # Create an empty array for the combined data
    combined_data = np.zeros((coad_avg_filtered.shape[0], 10))

    combined_data[:, 0::2] = coad_avg_filtered
    combined_data[:, 1::2] = normal_avg_filtered

    cell_types = ['Blood-B', 'Blood-Granul', 'Blood-Mono+Macro', 'Blood-NK', 'Blood-T',
                          'Bladder-Ep', 'Colon-Ep', 'Pancreas-Delta', 'Lung-Ep-Alveo', 'Gastric-Ep',
                          'Breast-Luminal-Ep', 'Neuron', 'Pancreas-Alpha', 'Oligodend', 'Smooth-Musc',
                          'Eryth-prog', 'Liver-Hep', 'Endothelium', 'Fallopian-Ep', 'Kidney-Ep',
                          'Skeletal-Musc', 'Pancreas-Acinar', 'Prostate-Ep', 'Pancreas-Beta',
                          'Heart-Fibro', 'Colon-Fibro', 'Head-Neck-Ep', 'Thyroid-Ep',
                          'Ovary+Endom-Ep', 'Adipocytes', 'Lung-Ep-Bron', 'Heart-Cardio',
                          'Pancreas-Duct', 'Breast-Basal-Ep', 'Small-Int-Ep']

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(combined_data, cmap='coolwarm', cbar=True)

    heatmap = sns.heatmap(combined_data, cmap='coolwarm', cbar=False, ax=ax)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_ticks_position('left')  # 将刻度线放在左侧
    cbar.ax.set_ylabel('', rotation=270, labelpad=15)  # 如果需要标签，可以在这里添加
    cbar.ax.tick_params(axis='y', labelsize=10, width=2, length=6)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.yticks(ticks=np.arange(0.5, len(cpg_labels) + 0.5), labels=cpg_labels, fontsize=12, fontweight='bold')

    # Add channel labels
    channel_labels = ['ALS_1', 'CTR_1', 'ALS_2', 'CTR_2', 'ALS_3', 'CTR_3', 'ALS_4', 'CTR_4', 'ALS_5', 'CTR_5']
    plt.xticks(ticks=np.arange(0.5, 10.5, 1), labels=channel_labels, fontsize=12, fontweight='bold')

    plt.xlabel('Channels', fontsize=12, fontweight='bold')
    plt.ylabel('CpG Sites (start-end)', fontsize=12, fontweight='bold')
    plt.title(f'Cell Type: {cell_types[cell_type]}, ALS vs CTR Average', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}/DTNN{cell_types[cell_type]}_{input_path1}_vs_{input_path2}_{fix}_Combined_Average.pdf')
    #plt.show()


def main():
    input_path1 = "ALS_WBC_35.txt"
    input_path2 = "ALS_CTR_WBC_35.txt"
    simi = 1
    mode = "high-resolution"
    n = 5000
    usenorm = "True"
    fix = "True"
    alpha = 2
    beta = 1
    for r1 in [[12, 64, 128]]:
        beta_est_path1 = './results/test' + input_path1 + str(simi) + mode + str(r1) + usenorm + str(n) + fix + str(alpha) + str(beta) + 'beta_est.pkl'
        with open(beta_est_path1, "rb") as fin:
            beta_est1 = pickle.load(fin)

        beta_est_path2 = './results/test' + input_path2 + str(simi) + mode + str(r1) + usenorm + str(n) + fix + str(alpha) + str(beta) + 'beta_est.pkl'
        with open(beta_est_path2, "rb") as fin:
            beta_est2 = pickle.load(fin)

        save_path = './train/results/heatmaps'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for cell_type in beta_est1.keys():
            coad_data = beta_est1[cell_type]
            normal_data = beta_est2[cell_type]

            coad_avg = coad_data.mean(axis=0)
            normal_avg = normal_data.mean(axis=0)

            output_path = f'./train/results/heatmaps/{input_path1}_{input_path2}_{fix}{alpha}{beta}_significant_top_50_cpgs1.bed'
            plot_combined_heatmaps(coad_avg, normal_avg, cell_type, save_path, r1, fix, input_path1, input_path2, output_path)

if __name__ == "__main__":
    main()

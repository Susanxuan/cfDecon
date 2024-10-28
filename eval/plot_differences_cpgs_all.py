import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_combined_heatmaps(unhealthy_avg, normal_avg, cell_type, save_path, r1, fix, input_path1, input_path2):
    # Create an empty array for the combined data
    combined_data = np.zeros((unhealthy_avg.shape[0], 10))

    # Interleave COAD and Normal data by channels
    combined_data[:, 0::2] = unhealthy_avg
    combined_data[:, 1::2] = normal_avg

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(combined_data, cmap='coolwarm', cbar=True)

    heatmap = sns.heatmap(combined_data, cmap='coolwarm', cbar=False, ax=ax)

    cbar = heatmap.collections[0].colorbar

    cbar.ax.yaxis.set_ticks_position('left')  # legend on left
    cbar.ax.set_ylabel('', rotation=270, labelpad=15)  # if label needed, add here

    cbar.ax.tick_params(axis='y', labelsize=10, width=2, length=6)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()
    cbar.ax.set_position([ax.get_position().x1+0.01,
                          ax.get_position().y0,
                          0.02,
                          ax.get_position().height])

    # cell_types = ['dendritic', 'endothelial', 'eosinophil', 'erythroblast', 'macrophage', 'monocyte', 'neutrophil',
    #               'placenta', 'tcell', 'adipose', 'brain', 'fibroblast', 'heart', 'hepatocyte', 'lung', 'mammary',
    #               'megakaryocyte', 'skeletal', 'small_intestine']

    cell_types = ['Blood-B', 'Blood-Granul', 'Blood-Mono+Macro', 'Blood-NK', 'Blood-T',
                          'Bladder-Ep', 'Colon-Ep', 'Pancreas-Delta', 'Lung-Ep-Alveo', 'Gastric-Ep',
                          'Breast-Luminal-Ep', 'Neuron', 'Pancreas-Alpha', 'Oligodend', 'Smooth-Musc',
                          'Eryth-prog', 'Liver-Hep', 'Endothelium', 'Fallopian-Ep', 'Kidney-Ep',
                          'Skeletal-Musc', 'Pancreas-Acinar', 'Prostate-Ep', 'Pancreas-Beta',
                          'Heart-Fibro', 'Colon-Fibro', 'Head-Neck-Ep', 'Thyroid-Ep',
                          'Ovary+Endom-Ep', 'Adipocytes', 'Lung-Ep-Bron', 'Heart-Cardio',
                          'Pancreas-Duct', 'Breast-Basal-Ep', 'Small-Int-Ep']

    # Add channel labels
    #channel_labels = ['COAD_0', 'Normal_0', 'COAD_1', 'Normal_1', 'COAD_2', 'Normal_2', 'COAD_3', 'Normal_3', 'COAD_4', 'Normal_4']
    channel_labels = ['ALS_0', 'CTR_0', 'ALS_1', 'CTR_1', 'ALS_2', 'CTR_2', 'ALS_3', 'CTR_3', 'ALS_4',
                      'CTR_4']

    plt.xticks(ticks=np.arange(0.5, 10.5, 1), labels=channel_labels)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel('Channels (Interleaved ALS and CTR)', fontsize=12, fontweight='bold')
    plt.ylabel('CpGs Positions', fontsize=12, fontweight='bold')
    plt.title(f'Cell Type: {cell_types[cell_type]}, ALS vs CTR Average', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{fix}_{cell_type}_{input_path1}_vs_{input_path2}_Combined_Average.pdf')
    #plt.show()

def main():
    input_path1 = "ALS_WBC_35.txt"  # ALS_WBC_35.txt "HOT_WBC_35.txt"
    input_path2 = "ALS_CTR_WBC_35.txt"  # ALS_CTR_WBC_35.txt "CTR_WBC_35.txt"
    simi = 1
    mode = "high-resolution"
    n = 5000
    usenorm = "True"
    fix = "True"
    alpha = 2
    beta = 1
    for r1 in [[12, 64, 128]]:  # [0], [150, 72, 48, 12], [150, 84, 52, 14]
        # beta_est_path1 = './results/test' + input_path1 + str(simi) + mode + str(r1) + 'beta_est.pkl'
        beta_est_path1 = './new_results/test' + input_path1 + str(simi) + mode + str(r1) + usenorm + str(n) + fix + str(
            alpha) + str(beta) + 'beta_est.pkl'


        with open(beta_est_path1, "rb") as fin:
            beta_est1 = pickle.load(fin)

        #beta_est_path2 = './results/test' + input_path2 + str(simi) + mode + str(r1) + 'beta_est.pkl'
        beta_est_path2 = './new_results/test' + input_path2 + str(simi) + mode + str(r1) + usenorm + str(n) + fix \
                         + str(alpha) + str(beta) + 'beta_est.pkl'
        with open(beta_est_path2, "rb") as fin:
            beta_est2 = pickle.load(fin)

        save_path = './results/heatmaps'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for cell_type in beta_est1.keys():
            unhealthy_data = beta_est1[cell_type]
            normal_data = beta_est2[cell_type]

            unhealthy_avg = unhealthy_data.mean(axis=0)
            normal_avg = normal_data.mean(axis=0)

            plot_combined_heatmaps(unhealthy_avg, normal_avg, cell_type, save_path, r1, fix, input_path1, input_path2)

if __name__ == "__main__":
    main()

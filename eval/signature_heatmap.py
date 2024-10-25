import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io as sio
from scipy.stats import spearmanr, pearsonr, ttest_ind, wilcoxon
import matplotlib

matplotlib.use('TkAgg')


def RMSEscore(pred, true):
    return np.mean(np.abs(pred - true))


def CCCscore(y_pred, y_true):
    ccc_value = 0
    for i in range(y_pred.shape[1]):
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        ccc_value += ccc
    return ccc_value / y_pred.shape[1]


def score(pred, label):
    distance = []
    ccc = []
    new_pred = pred.reshape(-1, 1)
    new_label = label.reshape(-1, 1)
    distance.append(RMSEscore(new_pred, new_label))
    ccc.append(CCCscore(new_pred, new_label))
    return distance[0], ccc[0]


def process_beta(beta_est):
    beta_est = beta_est.reshape(beta_est.shape[0], -1, beta_est.shape[2])
    beta_est[np.isnan(beta_est)] = 0
    sum_last_dim = np.sum(beta_est, axis=2, keepdims=True)
    beta_est = beta_est / sum_last_dim
    beta_est[np.isnan(beta_est)] = 0
    beta_est_sum = np.sum(beta_est, axis=0)  # 变为 m * 5

    return beta_est_sum


def calculate_scores(beta_est, beta_truth):
    all_results = []
    for i in range(beta_est.shape[0]):
        L1 = []
        CCC = []
        for j in range(beta_est.shape[2]):
            l1, ccc = score(beta_est[i, :, j], beta_truth[i, :, j])
            L1.append(l1)
            CCC.append(ccc)
        avg_l1 = sum(L1) / len(L1)
        avg_ccc = sum(CCC) / len(CCC)
        print(f"Samples {i}: Avg L1 = {avg_l1}, Avg CCC = {avg_ccc}")

        all_results.append(["Cell_Type", f"Cell_Type_{i}", avg_l1, avg_ccc])

    for i in range(beta_est.shape[2]):
        L1 = []
        CCC = []
        for j in range(beta_est.shape[0]):
            l1, ccc = score(beta_est[j, :, i], beta_truth[j, :, i])
            L1.append(l1)
            CCC.append(ccc)
        avg_l1 = sum(L1) / len(L1)
        avg_ccc = sum(CCC) / len(CCC)
        print(f"Channels {i}: Avg L1 = {avg_l1}, Avg CCC = {avg_ccc}")
        all_results.append(["Channel", f"Channel_{i}", avg_l1, avg_ccc])
    return all_results


def plot_scatter_plots(beta_est, beta_truth, r1, input_path, simi, mode, file):
    num_plots = beta_est.shape[0] * beta_est.shape[2]
    fig, axs = plt.subplots(beta_est.shape[0], beta_est.shape[2],
                            figsize=(beta_est.shape[2] * 3, beta_est.shape[0] * 3))
    cell_types = ['Blood-B', 'Blood-Granul', 'Blood-Mono+Macro', 'Blood-NK', 'Blood-T',
                  'Bladder-Ep', 'Colon-Ep', 'Pancreas-Delta', 'Lung-Ep-Alveo', 'Gastric-Ep',
                  'Breast-Luminal-Ep', 'Neuron', 'Pancreas-Alpha', 'Oligodend', 'Smooth-Musc',
                  'Eryth-prog', 'Liver-Hep', 'Endothelium', 'Fallopian-Ep', 'Kidney-Ep',
                  'Skeletal-Muscle', 'Pancreas-Acinar', 'Prostate-Ep', 'Pancreas-Beta',
                  'Heart-Fibro', 'Colon-Fibro', 'Head-Neck-Ep', 'Thyroid-Ep',
                  'Ovary+Endom-Ep', 'Adipocytes', 'Lung-Ep-Bron', 'Heart-Cardio',
                  'Pancreas-Duct', 'Breast-Basal-Ep', 'Small-Int-Ep']
    for i in range(beta_est.shape[0]):  # range(beta_est.shape[0]):
        for j in range(beta_est.shape[2]):
            x = beta_est[i, :, j]
            y = beta_truth[i, :, j]
            axs[i, j].scatter(x, y)
            ccc = CCCscore(x.reshape(-1, 1), y.reshape(-1, 1))
            axs[i, j].text(0.5, 0.9, f"CCC: {ccc:.3f}", ha='center', va='center', transform=axs[i, j].transAxes)
            axs[i, j].set_title(f"{cell_types[i]}, channel {j}")
            axs[i, j].set_xlim([0, 1])
            axs[i, j].set_ylim([0, 1])

    fig.tight_layout()
    # fig.savefig('./scatter_figure/baseline' + input_path + str(simi) + mode + file + '.png')
    # fig.savefig('./scatter_figure/' + input_path + str(simi) + mode + file + str(r1) + '.png')
    # fig.savefig('./scatter_figure/celfeer' + input_path + str(simi) + mode + file + '.png')
    # fig.savefig('./scatter_figure/cfsort' + input_path + str(simi) + mode + file + '.png')
    # plt.show()


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


def main():

    fix = "True"
    for r1 in [[12, 64, 128]]:
        simi = 1
        mode = "high-resolution"
        alpha, beta = 2, 1
        use_norm = "True"
        n = 5000
        input_path1 = "ALS_WBC_35.txt"
        input_path2 = "ALS_CTR_WBC_35.txt"
        # "HOT_WBC_35.txt", "CTR_WBC_35.txt", "TBR_WBC_35.txt", "ALS_WBC_35.txt", "ALS_CTR_WBC_35.txt"
        # for n in [1000, 2500, 5000, 7500]:
        # for file in ["0.1False0.4", "0.3False0.4", "0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"]:
        # print("Processing:", file)

        beta_est_path1 = f'./new_results/test{input_path1}{simi}{mode}{r1}{use_norm}{n}{fix}{alpha}{beta}beta_est.pkl'
        beta_est_path1 = f'./new_results/testcelfeer{input_path1}beta_est.pkl'

        with open(beta_est_path1, "rb") as fin:
            beta_est1 = pickle.load(fin)
        #beta_est1 = beta_est1[20]

        beta_est_path2 = f'./new_results/test{input_path2}{simi}{mode}{r1}{use_norm}{n}{fix}{alpha}{beta}beta_est.pkl'
        beta_est_path2 = f'./new_results/testcelfeer{input_path2}beta_est.pkl'
        with open(beta_est_path2, "rb") as fin:
            beta_est2 = pickle.load(fin)
        #beta_est2 = beta_est2[20]


        beta_est_sum1 = process_beta(beta_est1)
        beta_est_sum2 = process_beta(beta_est2)


        bed_file_path = f'./results/heatmaps/{input_path1}_{input_path2}_{fix}{alpha}{beta}_significant_top_50_cpgs1.bed'
        cpg_indices, cpg_labels = get_cpg_indices_and_labels(bed_file_path)

        # 提取指定 CpG 的 beta_est 数据
        selected_cpgs1 = beta_est_sum1[cpg_indices]  #
        selected_cpgs2 = beta_est_sum2[cpg_indices]  # CTR
        all_relative_values = []
        for channel in range(5):
            # 计算 CTR 和 HOT 的均值
            hot_mean = selected_cpgs1[:, channel]
            ctr_mean = selected_cpgs2[:, channel]

            relative_values = []

            for i in range(len(cpg_labels)):
                if ctr_mean[i] != 0:
                    relative_change = (hot_mean[i] - ctr_mean[i]) / ctr_mean[i]
                    relative_values.append(abs(relative_change))  # 取绝对值
                else:
                    relative_values.append(np.nan)  # 如果 CTR 为 0，设置为 NaN

            all_relative_values.append(relative_values)  # 存储每个通道的相对变化率

            relative_df = pd.DataFrame({
                'CpG Site': cpg_labels,
                'Relative Change': relative_values
            })

            # 绘制柱状图
            plt.figure(figsize=(10, 6))
            sns.barplot(x='CpG Site', y='Relative Change', data=relative_df, palette='coolwarm')
            plt.title(f'Absolute Relative Change for Channel {channel + 1}')
            plt.ylabel('Absolute Relative Change')
            plt.xlabel('CpG Sites')
            plt.xticks(rotation=90)
            plt.ylim(0, 4)
            plt.tight_layout()
            #plt.savefig(f'results/figure/cfdecon_{input_path1}_vs_{input_path2}_{fix}_channel{channel}_value.pdf')
            plt.savefig(f'results/figure/celfeer_{input_path1}_vs_{input_path2}_{fix}_channel{channel}_value.pdf')
            #plt.show()

        mean_relative_values = np.nanmean(all_relative_values, axis=0)


        mean_relative_df = pd.DataFrame({
            'CpG Site': cpg_labels,
            'Mean Absolute Relative Change': mean_relative_values
        })

        plt.figure(figsize=(10, 6))
        sns.barplot(x='CpG Site', y='Mean Absolute Relative Change', data=mean_relative_df, palette='coolwarm')
        plt.title('Mean Absolute Relative Change Across All Channels')
        plt.ylabel('Mean Absolute Relative Change')
        plt.xlabel('CpG Sites')
        plt.xticks(rotation=90)
        plt.ylim(0, 1)
        plt.tight_layout()
        #plt.savefig(f'results/figure/cfdecon_{input_path1}_vs_{input_path2}_{fix}_average_value.pdf')
        plt.savefig(f'results/figure/celfeer_{input_path1}_vs_{input_path2}_{fix}_average_value.pdf')
        #plt.show()


if __name__ == "__main__":
    main()

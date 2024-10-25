import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import scipy.io as sio


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


def process_beta(beta_truth, beta_est):
    beta_est = beta_est.reshape(beta_truth.shape[0], -1, beta_truth.shape[2])
    beta_est[np.isnan(beta_est)] = 0
    sum_last_dim = np.sum(beta_est, axis=2, keepdims=True)
    beta_est = beta_est / sum_last_dim
    beta_est[np.isnan(beta_est)] = 0
    return beta_est


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
        print(f"Cell Type {i}: Avg L1 = {avg_l1}, Avg CCC = {avg_ccc}")

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
        print(f"Channel {i}: Avg L1 = {avg_l1}, Avg CCC = {avg_ccc}")

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
    fig.savefig('./scatter_figure/baseline' + input_path + str(simi) + mode + file + '.pdf')
    # fig.savefig('./scatter_figure/' + input_path + str(simi) + mode + file + str(r1) + '.pdf')
    # fig.savefig('./scatter_figure/celfeer' + input_path + str(simi) + mode + file + '.pdf')
    # fig.savefig('./scatter_figure/cfsort' + input_path + str(simi) + mode + file + '.png')
    # plt.show()


def main():
    fix = "False" # used for UXM data
    for input_path in ["WGBS_ref_2+1.txt", "WGBS_ref_1+2.txt"]:
        all_files_results = []
        for r1 in [[128, 72, 48, 12]]:   #[12, 64, 128]
            #input_path = "WGBS_ref_2+1.txt"  # "WGBS_ref_2+1.txt" "WGBS_sim_input.txt" "UXM_ref35_100m.bed"
            simi = 1
            mode = "overall"
            alpha, beta = 10, 1
            use_norm = "False"
            for file in ["0.1False0.4", "0.3False0.4", "0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"]:
            #for n in [1000, 2500, 5000, 7500]:
                # for file in ["0.1False0.4", "0.3False0.4", "0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"]:
                beta_truth_path = f'./new_results/{input_path}{simi}{mode}{file}{r1}{use_norm}{alpha}{beta}beta_true.pkl'
                with open(beta_truth_path, "rb") as fin:
                    beta_truth = pickle.load(fin)
                beta_est_path = f'./new_results/{input_path}{simi}{mode}{file}{r1}{use_norm}{alpha}{beta}beta_est.pkl'
                with open(beta_est_path, "rb") as fin:
                    beta_est = pickle.load(fin)
                beta_est = process_beta(beta_truth, beta_est)
                file_results = calculate_scores(beta_est, beta_truth)
                for result in file_results:
                    all_files_results.append([file] + result)  # file str(n)
                plot_scatter_plots(beta_est, beta_truth, r1, input_path, simi, mode, file)  # file str(n) + fix
        df = pd.DataFrame(all_files_results, columns=["File", "Type", "Type_ID", "L1", "CCC"])
        df.to_excel(f"./metirc/signature{input_path}{simi}{mode}{r1}{fix}{alpha}{beta}results_L1_CCC.xlsx",
                    index=False)
        print("Results saved to results_L1_CCC_per_cell_type_and_channel.xlsx")


if __name__ == "__main__":
    main()


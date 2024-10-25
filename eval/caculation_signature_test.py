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
    #fig.savefig('./scatter_figure/' + input_path + str(simi) + mode + file + str(r1) + '.png')
    # plt.show()


def main():
    fix = "False"
    for fix in ["True"]:
        all_files_results = []
        for r1 in [[12, 64, 128]]:
            simi = 1
            mode = "overall"
            alpha, beta = 2, 1
            use_norm = "True"
            n = 5000
            for input_path in ["HOT_WBC_35.txt", "CTR_WBC_35.txt", "TBR_WBC_35.txt", "ALS_WBC_35.txt", "ALS_CTR_WBC_35.txt"]:
                beta_truth_path = f'./new_results/test{input_path}{simi}{mode}{r1}{use_norm}{n}{fix}{alpha}{beta}beta_true.pkl'
                with open(beta_truth_path, "rb") as fin:
                    beta_truth = pickle.load(fin)
                beta_est_path = f'./new_results/test{input_path}{simi}{mode}{r1}{use_norm}{n}{fix}{alpha}{beta}beta_est.pkl'
                with open(beta_est_path, "rb") as fin:
                    beta_est = pickle.load(fin)
                beta_est = process_beta(beta_truth, beta_est)
                file_results = calculate_scores(beta_est, beta_truth)
                for result in file_results:
                    all_files_results.append([input_path] + result)  # str(n)
                # plot_scatter_plots(beta_est, beta_truth, r1, input_path, simi, mode, fix)  # str(n) + fix
        df = pd.DataFrame(all_files_results, columns=["File", "Type", "Type_ID", "L1", "CCC"])
        df.to_excel(f"./metirc/signature{input_path}{simi}{mode}{r1}{fix}{alpha}{beta}results_L1_CCC.xlsx",
                   index=False)


if __name__ == "__main__":
    main()

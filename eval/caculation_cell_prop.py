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


def process_alpha(alpha_truth, alpha_est):
    alpha_est = alpha_est.reshape(alpha_truth.shape[0], -1, alpha_truth.shape[2])
    alpha_est[np.isnan(alpha_est)] = 0
    sum_last_dim = np.sum(alpha_est, axis=2, keepdims=True)
    alpha_est = alpha_est / sum_last_dim
    alpha_est[np.isnan(alpha_est)] = 0
    return alpha_est


def calculate_scores(alpha_est, alpha_truth):
    L1 = []
    CCC = []
    for i in range(alpha_est.shape[1]):
        l1, ccc = score(alpha_est[:, i], alpha_truth[:, i])
        L1.append(l1)
        CCC.append(ccc)
    return L1, CCC


def plot_scatter_plots(alpha_est, alpha_truth, r1, input_path, simi, mode, file):
    num_plots = alpha_est.shape[0] * alpha_est.shape[2]
    fig, axs = plt.subplots(alpha_est.shape[0], alpha_est.shape[2],
                            figsize=(alpha_est.shape[2] * 3, alpha_est.shape[0] * 3))

    for i in range(alpha_est.shape[0]):
        for j in range(alpha_est.shape[2]):
            x = alpha_est[i, :, j]
            y = alpha_truth[i, :, j]
            axs[i, j].scatter(x, y)
            axs[i, j].set_title(f"Cell type {i},One hot {j}")
            axs[i, j].set_xlim([0, 1])
            axs[i, j].set_ylim([0, 1])

    fig.tight_layout()
    fig.savefig('./results/figure/' + input_path + str(simi) + mode + file + str(r1) + '.png')
    # plt.show()


def main():
    all_results = []
    for r1 in [[12, 64, 128]]:  # [0], [128, 72, 48, 12], [128, 64, 12]
        input_path = "WGBS_ref_1+2.txt"  # "WGBS_ref_1+2.txt", "WGBS_ref_2+1.txt"  "UXM_ref35_100m.bed"
        # "WGBS_sim_input.txt"
        simi = 1
        mode = "overall"
        alpha, beta = 10, 1
        use_norm = "False"
        num_tissue = 9
        fix = "False"
        file = "0.1false0.4"
        for file in ["0.1False0.4", "0.3False0.4", "0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"]:
            # for n in [1000, 2500, 5000, 7500, 10000]:
            # file = str(n) + fix

            alpha_truth_path = './new_results/' + input_path + str(simi) + mode + file + str(r1) + use_norm + str(
                alpha) + str(beta) + 'alpha_true.pkl'

            with open(alpha_truth_path, "rb") as fin:
                alpha_truth = pickle.load(fin)

            alpha_est_path = './new_results/' + input_path + str(simi) + mode + file + str(r1) + use_norm + str(
                alpha) + str(beta) + 'alpha_est.pkl'
            with open(alpha_est_path, "rb") as fin:
                alpha_est = pickle.load(fin)
            L1, CCC = calculate_scores(alpha_est, alpha_truth.cpu().numpy()) #
            l1, ccc = score(alpha_est, alpha_truth.cpu().numpy()) #
            print("l1", l1, "ccc", ccc)
            num_cell_types = len(L1)
            for i in range(num_cell_types):
                all_results.append([file, f"Cell_Type_{i + 1}", L1[i], CCC[i]])
            all_results.append([file, "Overall", l1, ccc])
    df = pd.DataFrame(all_results, columns=["File", "Cell_Type", "L1", "CCC"])
    df.to_excel('./metirc/' + input_path + str(simi) + mode + str(r1) + file + use_norm + str(alpha) + str(
            beta) + "L1_CCC.xlsx", index=False)

    print("Results saved to results_L1_CCC.xlsx")


if __name__ == "__main__":
    main()

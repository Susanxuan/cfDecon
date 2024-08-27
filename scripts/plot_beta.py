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
    L1 = {}
    CCC = {}
    for i in range(beta_est.shape[0]):
        L1[str(i)] = []
        CCC[str(i)] = []
        for j in range(beta_est.shape[2]):
            l1, ccc = score(beta_est[i, :, j], beta_truth[i, :, j])
            L1[str(i)].append(l1)
            CCC[str(i)].append(ccc)
        print(str(i), ':', sum(L1[str(i)]) / len(L1[str(i)]), sum(CCC[str(i)]) / len(CCC[str(i)]))

    L1 = {}
    CCC = {}
    for i in range(beta_est.shape[2]):
        L1[str(i)] = []
        CCC[str(i)] = []
        for j in range(beta_est.shape[0]):
            l1, ccc = score(beta_est[j, :, i], beta_truth[j, :, i])
            L1[str(i)].append(l1)
            CCC[str(i)].append(ccc)
        print(str(i), ':', sum(L1[str(i)]) / len(L1[str(i)]), sum(CCC[str(i)]) / len(CCC[str(i)]))


def plot_scatter_plots(beta_est, beta_truth, r1, input_path, simi, mode, file):
    num_plots = beta_est.shape[0] * beta_est.shape[2]
    fig, axs = plt.subplots(beta_est.shape[0], beta_est.shape[2],
                            figsize=(beta_est.shape[2] * 3, beta_est.shape[0] * 3))

    for i in range(beta_est.shape[0]):
        for j in range(beta_est.shape[2]):
            x = beta_est[i, :, j]
            y = beta_truth[i, :, j]
            axs[i, j].scatter(x, y)
            axs[i, j].set_title(f"Cell type {i},One hot {j}")
            axs[i, j].set_xlim([0, 1])
            axs[i, j].set_ylim([0, 1])

    fig.tight_layout()
    # fig.savefig('./results/figure/' + str(r1) + 'scatter_plots.png')
    fig.savefig('./results/figure/new' + input_path + str(simi) + mode + file + str(r1) + '.png')


def main():
    for r1 in [[128, 72, 48, 12]]:  # [0], [128, 72, 48, 12], ([64, 48, 32, 8], [8, 32, 48, 64]),
        input_path = "ALS.txt"  # "WGBS_ref_1+2.txt", "WGBS_ref_2+1.txt" "UXM_39.bed"
        simi = 1
        mode = "overall"
        file = "0.3False0.4"
        alpha = 20
        beta = 1
        for file in ["0.1False0.4", "0.3False0.4", "0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"]:
            # (10, 10), (10, 20), (1, 20), (1, 10), (20, 20)
            # "0.1False0.4", "0.3False0.4", "0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"
            print(file, alpha, beta)
            print("r1", r1, "\n")
            beta_truth_path = './results/newtest' + input_path + str(simi) + mode + file + str(r1) + str(alpha) + str(beta)+ 'beta_true.pkl' #+str(alpha)+str(beta)
            # beta_truth_path = './results/' + 'beta_true.pkl'
            with open(beta_truth_path, "rb") as fin:
                beta_truth = pickle.load(fin)
            print('beta_truth', beta_truth.shape)

            beta_est_path = './results/newtest' + input_path + str(simi) + mode + file + str(r1) + str(alpha) + str(beta)+ 'beta_est.pkl'
            # beta_est_path = './results/' + 'beta_est.pkl'
            with open(beta_est_path, "rb") as fin:
                beta_est = pickle.load(fin)

            # print('beta_est', beta_est.shape)

            # alpha_est_path = './results/' + input_path + str(simi) + mode + file + str(r1) + 'alpha_est.pkl'
            # # beta_est_path = './results/' + 'beta_est.pkl'
            # with open(alpha_est_path, "rb") as fin:
            #     alpha_est = pickle.load(fin)

            # # 存x_recon for奇异值能量曲线
            # alpha_est_path = './results/test' + input_path + str(simi) + mode + file + str(r1) + 'alpha_est.pkl'
            # # beta_est_path = './results/' + 'beta_est.pkl'
            # with open(alpha_est_path, "rb") as fin:
            #     alpha_est = pickle.load(fin)
            # x_recon = torch.matmul(torch.from_numpy(alpha_est), torch.from_numpy(beta_est).permute(1, 0, 2))
            # x_recon_np_tensor = x_recon.numpy()
            # sio.savemat('./results/singular/x_recon_tensor_als.mat', {'x_recon_tensor_als': x_recon_np_tensor})
            # print("save successfully")

            beta_est = process_beta(beta_truth, beta_est)
            print('beta_est', beta_est.shape)

            calculate_scores(beta_est, beta_truth)
            plot_scatter_plots(beta_est, beta_truth, r1, input_path, simi, mode, file)


if __name__ == "__main__":
    main()

# 存x_recon for奇异值能量曲线
# alpha_est_path = './results/test' + input_path + str(simi) + mode + str(r1) + 'alpha_est.pkl'
# # beta_est_path = './results/' + 'beta_est.pkl'
# with open(alpha_est_path, "rb") as fin:
#     alpha_est = pickle.load(fin)
# x_recon = torch.matmul(torch.from_numpy(alpha_est), torch.from_numpy(beta_est).permute(1, 0, 2))
# x_recon_np_tape = x_recon.numpy()
# sio.savemat('./results/singular/x_recon_tape.mat', {'x_recon_tape': x_recon_np_tape})

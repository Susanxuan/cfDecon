import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


def RMSEscore(pred, true):
    return np.mean(np.abs(pred - true))


def CCCscore(y_pred, y_true):
    # pred: shape{n sample, m cell}
    ccc_value = 0
    for i in range(y_pred.shape[1]):
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        # print(r)
        # Mean
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        # Variance
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        # Standard deviation
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        # print(ccc)
        ccc_value += ccc
    return ccc_value / y_pred.shape[1]


def score(pred, label):
    distance = []
    ccc = []
    new_pred = pred.reshape(-1, 1)
    new_label = label.reshape(-1, 1)
    distance.append(RMSEscore(new_pred, new_label))
    ccc.append(CCCscore(new_pred, new_label))
    # print(distance[0], ccc[0])
    return distance[0], ccc[0]


def main():
    for r1 in [[128, 72, 48, 12]]:  # [128, 72, 48, 12], [128, 84, 52, 14]

        input_path = "WGBS_ref_1+2.txt"
        simi = 2
        mode = "high-resolution"
        # beta_truth_path = './results/' + input_path + str(simi) + str(r1) + 'beta_true.pkl'
        # # beta_truth_path = './results/' + 'beta_true.pkl'
        # with open(beta_truth_path, "rb") as fin:
        #     beta_truth = pickle.load(fin)
        # print('beta_truth', beta_truth.shape)

        # dir_name='new_data_output/Multi_channel_WGBS/one_sim_spar0.3_ref_1+2_high_2sim'
        # dir_name='new_data_output/Multi_channel_WGBS/two_sim_spar0.5_ref_1+2_h'

        # beta_truth_path_1='/mnt/nas/user/yixuan/cfDNA/CelFEER/output/'+dir_name+'/beta_true_1.pkl'
        for file in ["0.1False0.4", "0.3False0.4", "0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"]: #"0.1False0.4", "0.3False0.4", "0.3True0.1"
            # "0.1False0.4", "0.3False0.4", "0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3",
            print(file)
            print("r1", r1, "\n")
            beta_truth_path_1 = './results/' + input_path + str(simi) + mode + file + str(r1) + 'beta_true_1.pkl'

            with open(beta_truth_path_1, "rb") as fin:
                beta_truth_1 = pickle.load(fin)
            beta_truth_path_2 = './results/' + input_path + str(simi) + mode + file + str(r1) + 'beta_true_2.pkl'
            with open(beta_truth_path_2, "rb") as fin:
                beta_truth_2 = pickle.load(fin)

            beta_est_path = './results/' + input_path + str(simi) + mode + file + str(r1) + 'beta_est.pkl'
            with open(beta_est_path, "rb") as fin:
                beta_est = pickle.load(fin)

            # Euclidean distance
            euclidean_distance = euclidean(beta_truth_1.flatten(), beta_truth_2.flatten())
            print("Euclidean Distance:", euclidean_distance)

            # Cosine similarity
            cosine = cosine_similarity(beta_truth_1.reshape(1, -1), beta_truth_2.reshape(1, -1))
            print("Cosine Similarity:", cosine[0][0])

            # Correlation coefficient
            correlation_coefficient, _ = pearsonr(beta_truth_1.flatten(), beta_truth_2.flatten())
            print("Correlation Coefficient:", correlation_coefficient)

            # Frobenius norm
            frobenius_norm = np.linalg.norm(beta_truth_1 - beta_truth_2)
            print("Frobenius Norm:", frobenius_norm)

            L1 = {}
            CCC = {}
            L1_2 = {}
            CCC_2 = {}
            L1_cell_11 = {}
            CCC_cell_11 = {}
            L1_cell_12 = {}
            CCC_cell_12 = {}
            cell_compare_1 = []
            cell_compare_2 = []
            for c in beta_est.keys():
                sum_last_dim = np.sum(beta_est[c], axis=2, keepdims=True)
                beta_est[c] = beta_est[c] / sum_last_dim  # normalize
                beta_est[c][np.isnan(beta_est[c])] = 0
                L1[str(c)] = {}
                CCC[str(c)] = {}
                L1_2[str(c)] = {}
                CCC_2[str(c)] = {}
                L1_cell_11[str(c)] = []
                CCC_cell_11[str(c)] = []
                L1_cell_12[str(c)] = []
                CCC_cell_12[str(c)] = []
                for i in range(int(beta_est[c].shape[0] / 2)):
                    L1[str(c)][str(i)] = []
                    CCC[str(c)][str(i)] = []
                    L1_2[str(c)][str(i)] = []
                    CCC_2[str(c)][str(i)] = []
                    ii = i
                    temp_beta_est = beta_est[c][ii, :, :].reshape(1, beta_est[c].shape[1], beta_est[c].shape[2])
                    for j in range(beta_est[c].shape[2]):
                        l1, ccc = score(temp_beta_est[:, :, j], beta_truth_1[c, :, j])
                        l1_2, ccc_2 = score(temp_beta_est[:, :, j], beta_truth_2[c, :, j])
                        L1[str(c)][str(i)].append(l1)
                        CCC[str(c)][str(i)].append(ccc)
                        L1_2[str(c)][str(i)].append(l1_2)
                        CCC_2[str(c)][str(i)].append(ccc_2)
                    L1_cell_11[str(c)].append(sum(L1[str(c)][str(i)]) / len(L1[str(c)][str(i)]))
                    CCC_cell_11[str(c)].append(sum(CCC[str(c)][str(i)]) / len(CCC[str(c)][str(i)]))
                    L1_cell_12[str(c)].append(sum(L1_2[str(c)][str(i)]) / len(L1_2[str(c)][str(i)]))
                    CCC_cell_12[str(c)].append(sum(CCC_2[str(c)][str(i)]) / len(CCC_2[str(c)][str(i)]))
                cell_compare_1.append(sum(CCC_cell_11[str(c)]) / len(CCC_cell_11[str(c)]))
                cell_compare_2.append(sum(CCC_cell_12[str(c)]) / len(CCC_cell_12[str(c)]))
                print('cell type ', str(c), 'simulation 1 CCC with reference 1:',
                      sum(CCC_cell_11[str(c)]) / len(CCC_cell_11[str(c)]))
                print('cell type ', str(c), 'simulation 1 CCC with reference 2:',
                      sum(CCC_cell_12[str(c)]) / len(CCC_cell_12[str(c)]))
            print('simulation 1 CCC with reference 1:', sum(cell_compare_1) / len(cell_compare_1))
            print('simulation 1 CCC with reference 2:', sum(cell_compare_2) / len(cell_compare_2))

            L1 = {}
            CCC = {}
            L1_2 = {}
            CCC_2 = {}
            L1_cell_11 = {}
            CCC_cell_11 = {}
            L1_cell_12 = {}
            CCC_cell_12 = {}
            cell_compare_1 = []
            cell_compare_2 = []
            for c in beta_est.keys():  # cell_type
                sum_last_dim = np.sum(beta_est[c], axis=2, keepdims=True)
                beta_est[c] = beta_est[c] / sum_last_dim  # normalize
                beta_est[c][np.isnan(beta_est[c])] = 0
                L1[str(c)] = {}
                CCC[str(c)] = {}
                L1_2[str(c)] = {}
                CCC_2[str(c)] = {}
                L1_cell_11[str(c)] = []
                CCC_cell_11[str(c)] = []
                L1_cell_12[str(c)] = []
                CCC_cell_12[str(c)] = []
                for i in range(int(beta_est[c].shape[0] / 2)):  # 1+2
                    L1[str(c)][str(i)] = []
                    CCC[str(c)][str(i)] = []
                    L1_2[str(c)][str(i)] = []
                    CCC_2[str(c)][str(i)] = []
                    ii = i + int(beta_est[c].shape[0] / 2)
                    # ii = i
                    temp_beta_est = beta_est[c][ii, :, :].reshape(1, beta_est[c].shape[1], beta_est[c].shape[2])
                    for j in range(beta_est[c].shape[2]):
                        l1, ccc = score(temp_beta_est[:, :, j], beta_truth_1[c, :, j])
                        l1_2, ccc_2 = score(temp_beta_est[:, :, j], beta_truth_2[c, :, j])
                        L1[str(c)][str(i)].append(l1)
                        CCC[str(c)][str(i)].append(ccc)
                        L1_2[str(c)][str(i)].append(l1_2)
                        CCC_2[str(c)][str(i)].append(ccc_2)
                    L1_cell_11[str(c)].append(sum(L1[str(c)][str(i)]) / len(L1[str(c)][str(i)]))
                    CCC_cell_11[str(c)].append(sum(CCC[str(c)][str(i)]) / len(CCC[str(c)][str(i)]))
                    L1_cell_12[str(c)].append(sum(L1_2[str(c)][str(i)]) / len(L1_2[str(c)][str(i)]))
                    CCC_cell_12[str(c)].append(sum(CCC_2[str(c)][str(i)]) / len(CCC_2[str(c)][str(i)]))
                cell_compare_1.append(sum(CCC_cell_11[str(c)]) / len(CCC_cell_11[str(c)]))
                cell_compare_2.append(sum(CCC_cell_12[str(c)]) / len(CCC_cell_12[str(c)]))
                print('cell type ', str(c), 'simulation 2 CCC with reference 1:',
                      sum(CCC_cell_11[str(c)]) / len(CCC_cell_11[str(c)]))
                print('cell type ', str(c), 'simulation 2 CCC with reference 2:',
                      sum(CCC_cell_12[str(c)]) / len(CCC_cell_12[str(c)]))
            print('simulation 2 CCC with reference 1:', sum(cell_compare_1) / len(cell_compare_1))
            print('simulation 2 CCC with reference 2:', sum(cell_compare_2) / len(cell_compare_2))




if __name__ == "__main__":
    main()



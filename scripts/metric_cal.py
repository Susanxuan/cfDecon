import pickle
import numpy as np
import pandas as pd


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
    new_pred = pred.reshape(-1,1)
    new_label = label.reshape(-1,1)
    distance.append(RMSEscore(new_pred, new_label))
    ccc.append(CCCscore(new_pred, new_label))
    # print(distance[0], ccc[0])
    return distance[0], ccc[0]

# dir_name='WGBS_sim/1000_7_spar0.3'
# dir_name='WGBS_sim/2sims_test'
dir_name='WGBS_sim/two_sim_spar0.5_ref_1+2'
# dir_name='WGBS_sim/1000_10_spar0.3_ref_1+2'
# dir_name='generated_sim'
# dir_name='Multi_channel_AE_sim'

alpha_truth_path='/mnt/nas/user/yixuan/cfDNA/CelFEER/new_data_output/'+dir_name+'/1_alpha_true.pkl'
with open(alpha_truth_path, "rb") as fin:
    alpha_truth = pickle.load(fin)

print('alpha_truth',alpha_truth)

alpha_est_path='/mnt/nas/user/yixuan/cfDNA/CelFEER/new_data_output/'+dir_name+'/1_alpha_est.pkl'
with open(alpha_est_path, "rb") as fin:
    alpha_est = pickle.load(fin)

print('alpha_est',alpha_est)

L1 = []
CCC = []

for i in range(alpha_est.shape[0]):
    l1, ccc = score(alpha_est[i],alpha_truth[i])
    L1.append(l1)
    CCC.append(ccc)
    # print(l1, ccc)

print('all:',sum(L1)/len(L1),sum(CCC)/len(CCC))

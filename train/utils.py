import numpy as np
import matplotlib.pyplot as plt
import random

def showloss(loss,path):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig(path)

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
    return distance[0], ccc[0]


def generate_sample_array(n, c, sparse=False, sparse_prob=0.3, rare=False, rare_percentage=0.4, WBC_major=False,
                          UXM=False, fix_c=False):
    if WBC_major:
        print(
            'This simulation mode follows cfSort.'
            'The input reference for this mode is 5 Blood cells + c (c in range(1,10)) random cell types,'
            'You will set the sum of WBC cell type fractions to be in 70% to 80%.')
        p = np.random.uniform(0.7, 0.9, (n, 1))
        # Generate random values for Blood cells in the array
        blood_values = np.random.rand(n, 5)
        blood_row_sums = np.sum(blood_values, axis=1)
        blood_prop = blood_values / blood_row_sums[:, np.newaxis]

        # Generate random values for the extra cell in the array
        other_values = np.random.rand(n, c)
        # Normalize the values to ensure the sum of each row is 1
        other_row_sums = np.sum(other_values, axis=1)
        other_prop = other_values / other_row_sums[:, np.newaxis]

        # 将两个 proportions 分别乘以 p 和 (1-p)
        blood_prop_p = blood_prop * p
        other_prop_1_p = other_prop * (1 - p)

        # 将两个 proportions 拼接成一个新的 proportions
        prop = np.concatenate((blood_prop_p, other_prop_1_p), axis=1)
        # print(np.sum(prop, axis=1)) # check sum = 1
    elif UXM:
        print(
            'This simulation mode follows cfSort.'
            'The input reference for this mode is 5 Blood cells + 26 remaining cell types in UXM,'
            'You will set the sum of WBC cell type fractions to be in 70% to 80%.')
        assert c == 35, 'cell type numers error'
        p = np.random.uniform(0.7, 0.9, (n, 1))
        # Generate random values for Blood cells in the array
        blood_values = np.random.rand(n, 5)
        blood_row_sums = np.sum(blood_values, axis=1)
        blood_prop = blood_values / blood_row_sums[:, np.newaxis]

        # Generate random values for the extra cell in the array
        other_values = np.random.rand(n, c - 5)
        # Normalize the values to ensure the sum of each row is 1
        other_row_sums = np.sum(other_values, axis=1)
        other_prop = other_values / other_row_sums[:, np.newaxis]

        for i in range(n):
            if fix_c:
                indices = np.random.choice(np.arange(other_prop.shape[1]), replace=False, size=c - 5 - 9)
            else:
                indices = np.random.choice(np.arange(other_prop.shape[1]), replace=False,
                                           size=random.randint(c - 5 - 9, c - 5 - 1))
            #print(indices)
            other_prop[i, indices] = 0

        other_prop = other_prop / np.sum(other_prop, axis=1).reshape(-1, 1)

        # 将两个 proportions 分别乘以 p 和 (1-p)
        blood_prop_p = blood_prop * p
        other_prop_1_p = other_prop * (1 - p)

        # 将两个 proportions 拼接成一个新的 proportions
        prop = np.concatenate((blood_prop_p, other_prop_1_p), axis=1)
        # print(np.sum(prop, axis=1))  # check sum = 1
    else:
        # Generate random values for each cell in the array
        values = np.random.rand(n, c)

        # Normalize the values to ensure the sum of each row is 1
        row_sums = np.sum(values, axis=1)
        prop = values / row_sums[:, np.newaxis]

        if sparse:
            print("You set sparse as True, some cell's fraction will be zero, the probability is", sparse_prob)
            ## Only partial simulated data is composed of sparse celltype distribution
            for i in range(int(prop.shape[0] * sparse_prob)):
                indices = np.random.choice(np.arange(prop.shape[1]), replace=False,
                                           size=int(prop.shape[1] * sparse_prob))
                prop[i, indices] = 0

            prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

        if rare:
            print(
                'You will set some cell type fractions are very small (<3%), '
                'these celltype is randomly chosen by percentage you set before.')
            ## choose celltype
            np.random.seed(0)
            indices = np.random.choice(np.arange(prop.shape[1]), replace=False,
                                       size=int(prop.shape[1] * rare_percentage))
            prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

            for i in range(int(0.5 * prop.shape[0]) + int(int(rare_percentage * 0.5 * prop.shape[0]))):
                prop[i, indices] = np.random.uniform(0, 0.03, len(indices))
                buf = prop[i, indices].copy()
                prop[i, indices] = 0
                prop[i] = (1 - np.sum(buf)) * prop[i] / np.sum(prop[i])
                prop[i, indices] = buf

    return prop
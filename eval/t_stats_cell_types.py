import numpy as np
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # 导入seaborn库
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample


def permutation_test(data1, data2, n_permutations=10000):
    original_stat = np.mean(data1) - np.mean(data2)

    combined = np.concatenate([data1, data2])

    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_data1 = combined[:len(data1)]
        perm_data2 = combined[len(data1):]
        perm_stat = np.mean(perm_data1) - np.mean(perm_data2)
        if abs(perm_stat) >= abs(original_stat):
            count += 1

    p_value = count / n_permutations
    return original_stat, p_value


# Function to calculate AUC and 95% CI using bootstrapping
def bootstrap_auc(y_true, y_scores, n_bootstraps=1000, random_seed=42):
    np.random.seed(random_seed)
    bootstrapped_aucs = []

    for i in range(n_bootstraps):
        # Resample with replacement
        indices = resample(np.arange(len(y_scores)), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            # Skip this sample because it does not have both classes
            continue
        fpr, tpr, _ = roc_curve(y_true[indices], y_scores[indices])
        roc_auc = auc(fpr, tpr)
        bootstrapped_aucs.append(roc_auc)

    # Calculate 95% CI
    sorted_aucs = np.sort(bootstrapped_aucs)
    lower_ci = sorted_aucs[int(0.025 * len(sorted_aucs))]
    upper_ci = sorted_aucs[int(0.975 * len(sorted_aucs))]

    return lower_ci, upper_ci


for r1 in [[12, 64, 128]]:  # [12, 64, 128], [36, 128], [12, 48, 72, 128]
    input_path1 = "ALS_WBC_35.txt"  # "HOT_WBC_35.txt" "ALS_WBC_35.txt"
    input_path2 = "ALS_CTR_WBC_35.txt" # "CTR_WBC_35.txt" "ALS_CTR_WBC_35.txt"
    simi = 1
    mode = "overall"
    # file = "0.3True0.2"  # "0.3True0.1", "0.3True0.2", "0.3True0.3"
    n = 5000
    fix = "True"
    usenorm = "True"
    for alpha, beta in [(2, 1)]:
        alpha_est_path1 = './new_results/test' + input_path1 + str(simi) + mode + str(
            r1) + usenorm + str(n) + fix + str(alpha) + str(beta) + 'alpha_est.pkl'
        method = "celfeer"
        #alpha_est_path1 = './new_results/testcfsort' + input_path1 + str(simi) + mode + fix + str(n) + fix + 'alpha_est.pkl'
        #alpha_est_path1 = './new_results/testcelfeer' + input_path1 + 'alpha_est.pkl'
        with open(alpha_est_path1, "rb") as fin:
            alpha_est1 = pickle.load(fin)

        alpha_est_path2 = './new_results/test' + input_path2 + str(simi) + mode + str(
            r1) + usenorm + str(n) + fix + str(alpha) + str(beta) + 'alpha_est.pkl'
        #alpha_est_path2 = './new_results/testcfsort' + input_path2 + str(simi) + mode + fix + str(n) + fix + 'alpha_est.pkl'
        #alpha_est_path2 = './new_results/testcelfeer' + input_path2 + 'alpha_est.pkl'
        with open(alpha_est_path2, "rb") as fin:
            alpha_est2 = pickle.load(fin)

        cell_types1 = ['Blood-B', 'Blood-Granul', 'Blood-Mono+Macro', 'Blood-NK', 'Blood-T',
              'Bladder-Ep', 'Colon-Ep', 'Pancreas-Delta', 'Lung-Ep-Alveo', 'Gastric-Ep',
              'Breast-Luminal-Ep', 'Neuron', 'Pancreas-Alpha', 'Oligodend', 'Smooth-Musc',
              'Eryth-prog', 'Liver-Hep', 'Endothelium', 'Fallopian-Ep', 'Kidney-Ep',
              'Skeletal-Musc', 'Pancreas-Acinar', 'Prostate-Ep', 'Pancreas-Beta',
              'Heart-Fibro', 'Colon-Fibro', 'Head-Neck-Ep', 'Thyroid-Ep',
              'Ovary+Endom-Ep', 'Adipocytes', 'Lung-Ep-Bron', 'Heart-Cardio',
              'Pancreas-Duct', 'Breast-Basal-Ep', 'Small-Int-Ep']
        print("alpha", alpha, "beta", beta, r1)
        #  (Student's t-test)
        for i in range(35):
            t_stat, p_value = stats.ttest_ind(alpha_est1[:, i], alpha_est2[:, i], equal_var=False)
            print(f"T-statistic: {t_stat}, P-value: {p_value}, Cell Type: {cell_types1[i]}")

        #(Wilcoxon rank sum test)
        for i in range(35):
            w_stat, p_value = stats.ranksums(alpha_est1[:, i], alpha_est2[:, i])
            print(f"Wilcoxon statistic: {w_stat}, P-value: {p_value}, Cell Type: {cell_types1[i]}")

        # for i in range(35):
        #     u_stat, p_value = stats.mannwhitneyu(alpha_est1[:, i], alpha_est2[:, i])
        #     print(f"Mann-Whitney U statistic: {u_stat}, P-value: {p_value}, Cell Type: {cell_types1[i]}")
        #
        # for i in range(35):
        #     perm_stat, p_value = permutation_test(alpha_est1[:, i], alpha_est2[:, i])
        #     print(f"Permutation test statistic: {perm_stat}, P-value: {p_value}, Cell Type: {cell_types1[i]}")


        for i in range(35):
            mean_hot = np.mean(alpha_est1[:, i])
            std_hot = np.std(alpha_est1[:, i])
            mean_ctr = np.mean(alpha_est2[:, i])
            std_ctr = np.std(alpha_est2[:, i])

            print(f"Cell Type: {cell_types1[i]}")
            print(f"  ALS - Mean: {mean_hot:.4f}, Std: {std_hot:.4f}")
            print(f"  CTR - Mean: {mean_ctr:.4f}, Std: {std_ctr:.4f}")
            print("-" * 50)


        cell_types = ['Blood-B', 'Blood-Granul', 'Blood-Mono+Macro', 'Blood-NK', 'Blood-T',
                      'Bladder-Ep', 'Colon-Ep', 'Pancreas-Delta', 'Lung-Ep-Alveo', 'Gastric-Ep',
                      'Breast-Luminal-Ep', 'Neuron', 'Pancreas-Alpha', 'Oligodend', 'Smooth-Musc',
                      'Eryth-prog', 'Liver-Hep', 'Endothelium', 'Fallopian-Ep', 'Kidney-Ep',
                      'Skeletal-Musc', 'Pancreas-Acinar', 'Prostate-Ep', 'Pancreas-Beta',
                      'Heart-Fibro', 'Colon-Fibro', 'Head-Neck-Ep', 'Thyroid-Ep',
                      'Ovary+Endom-Ep', 'Adipocytes', 'Lung-Ep-Bron', 'Heart-Cardio',
                      'Pancreas-Duct', 'Breast-Basal-Ep', 'Small-Int-Ep']

        labels = np.concatenate([np.ones(alpha_est1.shape[0]), np.zeros(alpha_est2.shape[0])])
        # Example for a single cell type (e.g. 16th cell type)
        for i in [16]:
            scores = np.concatenate([alpha_est1[:, i], alpha_est2[:, i]])

            # Calculate FPR, TPR, and AUC
            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)

            # Calculate 95% CI for AUC using bootstrapping
            lower_ci, upper_ci = bootstrap_auc(labels, scores)

            # Plot ROC curve
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:0.2f} [95% CI: {lower_ci:0.2f}-{upper_ci:0.2f}])')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

            # Axis labels and limits
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (Specificity)')
            plt.ylabel('True Positive Rate (Sensitivity)')
            plt.title(f'ROC Curve for {cell_types[i]}')
            plt.legend(loc="upper left")

            # Save figure
            # plt.savefig(f'./auc/auc_{method}{input_path1}_vs_{input_path2}_{cell_types[i]}.pdf')
            # plt.show()






import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


# 全局选200/300

def calculate_weighted_difference(coad_avg, normal_avg, weights):
    differences = np.sqrt(np.sum(weights * (coad_avg - normal_avg) ** 2, axis=1))
    return differences


def save_all_top_cpgs_as_bed(data_df, all_top_cpgs_indices, output_path):
    with open(output_path, 'w') as bed_file:
        for idx in sorted(all_top_cpgs_indices):
            row = data_df.iloc[idx]
            chr, start, end = row[0], row[1], row[2]
            bed_file.write(f'{chr}\t{start}\t{end}\n')


def main():
    input_path1 = "ALS.txt"
    input_path2 = "ALS.txt"
    simi = 1
    mode = "high-resolution"
    weight_combinations = [
        [2.5, 1, 1, 2, 3.5]
    ]
    alpha_weight = [10]
    file = "0.3false0.4"
    alpha = 20
    beta = 1

    for r1 in [[128, 72, 48, 12]]:
        beta_est_path1 = './results/test' + input_path1 + str(simi) + mode + file + str(r1) + str(alpha) + str(
            beta) + 'beta_est.pkl'
        with open(beta_est_path1, "rb") as fin:
            beta_est1 = pickle.load(fin)

        beta_est_path2 = './results/test' + input_path2 + str(simi) + mode + file + str(r1) + str(alpha) + str(
            beta) + 'beta_est.pkl'
        with open(beta_est_path2, "rb") as fin:
            beta_est2 = pickle.load(fin)

        save_path = './results/heatmaps'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        input_path = "ALS.txt"
        data_df = pd.read_csv(input_path, delimiter="\t", header=None, skiprows=1)

        combined_differences = np.zeros((data_df.shape[0],))

        for weights, alpha in zip(weight_combinations, alpha_weight):
            for cell_type in beta_est1.keys():
                coad_data = beta_est1[cell_type][:4, :, :]
                normal_data = beta_est2[cell_type][4:, :, :]

                coad_avg = coad_data.mean(axis=0)
                normal_avg = normal_data.mean(axis=0)

                differences = calculate_weighted_difference(coad_avg, normal_avg, weights)
                combined_differences += differences

        # 获取差异最大的全局前200个CPG位点的索引
        top_global_cpgs_indices = np.argsort(combined_differences)[-200:][::-1]

        # 保存到文件
        output_path = './results/heatmaps/als_all_top_200_cpgs.bed'
        save_all_top_cpgs_as_bed(data_df, top_global_cpgs_indices, output_path)


if __name__ == "__main__":
    main()

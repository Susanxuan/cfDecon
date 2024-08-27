import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def calculate_weighted_difference(coad_avg, normal_avg, weights):
    differences = np.sqrt(np.sum(weights * (coad_avg - normal_avg) ** 2, axis=1))
    return differences


def save_all_top_cpgs_as_bed(data_df, all_top_cpgs_indices, output_path):
    with open(output_path, 'w') as bed_file:
        for weight_str, indices in all_top_cpgs_indices.items():
            # Use a set to avoid duplicate entries
            seen_indices = set()
            sorted_indices = sorted(indices)  # Sort indices if needed
            for idx in sorted_indices:
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    row = data_df.iloc[idx]
                    chr, start, end = row[0], row[1], row[2]
                    bed_file.write(f'{chr}\t{start}\t{end}\n')  # Removed \t{weight_str}



def main():
    input_path1 = "COAD_ALS.txt"
    input_path2 = "Normal_ALS.txt"
    simi = 1
    mode = "high-resolution"         # [2, 0.5, 0.5, 1.5, 4.5]
    weight_combinations = [
        [2.5, 1, 1, 2, 3.5]
        # [1, 0, 0, 0, 0],
        # [0, 1, 0, 0, 0],
        # [0, 0, 1, 0, 0],
        # [0, 0, 0, 1, 0],
        # [0, 0, 0, 0, 1]
    ]
    alpha_weight = [10] # [2.5, 1, 1, 2, 3.5]
    for r1 in [[128, 72, 48, 12]]:
        beta_est_path1 = './results/test' + input_path1 + str(simi) + mode + str(r1) + 'beta_est.pkl'
        with open(beta_est_path1, "rb") as fin:
            beta_est1 = pickle.load(fin)

        beta_est_path2 = './results/test' + input_path2 + str(simi) + mode + str(r1) + 'beta_est.pkl'
        with open(beta_est_path2, "rb") as fin:
            beta_est2 = pickle.load(fin)

        save_path = './results/heatmaps'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        input_path = "Bowel_ALS.txt"
        data_df = pd.read_csv(input_path, delimiter="\t", header=None, skiprows=1)  # read input samples/reference data

        results = {}

        for weights, alpha in zip(weight_combinations, alpha_weight):
            ratio = float(alpha/10)
            weight_str = ''.join(map(str, weights))
            results[weight_str] = {}

            for cell_type in [17, 11, 3, 15]:  # beta_est1.keys()
                coad_data = beta_est1[cell_type]
                normal_data = beta_est2[cell_type]

                coad_avg = coad_data.mean(axis=0)
                normal_avg = normal_data.mean(axis=0)

                differences = calculate_weighted_difference(coad_avg, normal_avg, weights)

                # 获取差异最大的前400个CPG位点的索引
                top_400_diff_indices = np.argsort(differences)[-int(ratio*400):][::-1]

                results[weight_str][cell_type] = top_400_diff_indices

        # Combine top 400 CpGs for all cell types
        all_top_cpgs_indices = {}
        for weights in weight_combinations:
            weight_str = ''.join(map(str, weights))
            all_top_cpgs_indices[weight_str] = []
            for cell_type in [17, 11, 3, 15]:  # beta_est1.keys()
                top_cpgs_indices = results[weight_str][cell_type]
                all_top_cpgs_indices[weight_str].extend(top_cpgs_indices)

        # Save combined top CpGs to a single .bed file
        output_path = './results/heatmaps/COAD_significant_top_400_cpgs.bed'
        save_all_top_cpgs_as_bed(data_df, all_top_cpgs_indices, output_path)

        # # 计算不同权重组合下前400个CpG位点的重合情况
        # overlap_data = {}
        # for cell_type in beta_est1.keys():
        #     overlap_counts = {}
        #     base_indices = results['11111'][cell_type]
        #     for weights in weight_combinations[1:]:
        #         weight_str = ''.join(map(str, weights))
        #         comparison_indices = results[weight_str][cell_type]
        #         overlap_count = len(set(base_indices).intersection(set(comparison_indices)))
        #         overlap_counts[weight_str] = overlap_count
        #
        #     overlap_data[cell_type] = overlap_counts
        #
        # # 绘制柱状图
        # fig, ax = plt.subplots(figsize=(400, 8))
        #
        # channel_names = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5']
        # cell_types = list(overlap_data.keys())
        #
        # overlap_matrix = np.array([[overlap_data[cell_type][wc] for wc in overlap_data[cell_type]] for cell_type in cell_types])
        # channel_sums = np.sum(overlap_matrix, axis=0)
        # bar_width = 0.15
        # index = np.arange(len(cell_types))
        #
        # for i in range(len(channel_names)):
        #     plt.bar(index + i * bar_width, overlap_matrix[:, i], bar_width, label=channel_names[i])
        #
        # plt.gca().spines['top'].set_linewidth(2)
        # plt.gca().spines['right'].set_linewidth(2)
        # plt.gca().spines['bottom'].set_linewidth(2)
        # plt.gca().spines['left'].set_linewidth(2)
        # plt.xticks(fontsize=12, fontweight='bold')
        # plt.yticks(fontsize=12, fontweight='bold')
        # plt.xlabel('Cell Types', fontsize=12, fontweight='bold')
        # plt.ylabel('Overlap Count', fontsize=12, fontweight='bold')
        # plt.title('Overlap Counts for Different Channels Across Cell Types', fontsize=16, fontweight='bold')
        # plt.xticks(index + bar_width * 2, cell_types)
        # plt.legend(prop={'weight': 'bold'})
        # plt.tight_layout()
        # plt.savefig('./results/figure/overlap_cpgs' + '.png')
        # #plt.show()


if __name__ == "__main__":
    main()

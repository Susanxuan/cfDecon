import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def calculate_weighted_difference(unhealthy_avg, normal_avg, weights):
    differences = np.sqrt(np.sum(weights * (unhealthy_avg - normal_avg) ** 2, axis=1))
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
    input_path1 = "ALS_WBC_35.txt"  # "ALS_WBC_35.txt" "HOT_WBC_35.txt"
    input_path2 = "ALS_CTR_WBC_35.txt"  # ALS_CTR_WBC_35.txt "CTR_WBC_35.txt"
    simi = 1
    mode = "high-resolution"
    weight_combinations = [
        # [1.5, 1, 1, 1.5, 2],
        [1, 1, 1, 1, 1],
        # [1, 0, 0, 0, 0],
        # [0, 1, 0, 0, 0],
        # [0, 0, 1, 0, 0],
        # [0, 0, 0, 1, 0],
        # [0, 0, 0, 0, 1]
    ]
    alpha = 2
    beta = 1
    usenorm = "True"
    fix = "True"
    n = 5000
    for r1 in [[12, 64, 128]]:
        beta_est_path1 = './new_results/test' + input_path1 + str(simi) + mode + str(r1) + usenorm + str(n) + fix + str(
            alpha) + str(beta) + 'beta_est.pkl'
        with open(beta_est_path1, "rb") as fin:
            beta_est1 = pickle.load(fin)
        beta_est_path2 = './new_results/test' + input_path2 + str(simi) + mode + str(r1) + usenorm + str(n) + fix \
                         + str(alpha) + str(beta) + 'beta_est.pkl'
        with open(beta_est_path2, "rb") as fin:
            beta_est2 = pickle.load(fin)
        save_path = './results/heatmaps'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        input_path = "data/ALS_WBC_35.txt"
        data_df = pd.read_csv(input_path, delimiter="\t", header=None, skiprows=1)

        results = {}
        for weights in weight_combinations:
            weight_str = ''.join(map(str, weights))
            results[weight_str] = {}

            for cell_type in [20]:  # beta_est1.keys()
                unhealthy_data = beta_est1[cell_type]
                normal_data = beta_est2[cell_type]
                unhealthy_avg = unhealthy_data.mean(axis=0)
                normal_avg = normal_data.mean(axis=0)
                differences = calculate_weighted_difference(unhealthy_avg, normal_avg, weights)
                top_50_diff_indices = np.argsort(differences)[-int(50):][::-1]
                results[weight_str][cell_type] = top_50_diff_indices

        # Combine top 50 CpGs for all cell types
        all_top_cpgs_indices = {}
        for weights in weight_combinations:
            weight_str = ''.join(map(str, weights))
            all_top_cpgs_indices[weight_str] = []
            for cell_type in [20]:  # 20 skeletal-muscle 16 liver-hep
                top_cpgs_indices = results[weight_str][cell_type]
                all_top_cpgs_indices[weight_str].extend(top_cpgs_indices)

        output_path = f'./results/heatmaps/{input_path1}_{input_path2}_{fix}{alpha}{beta}_significant_top_50_cpgs1.bed'
        save_all_top_cpgs_as_bed(data_df, all_top_cpgs_indices, output_path)

        # # Overlap
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
        # fig, ax = plt.subplots(figsize=(100, 8))
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

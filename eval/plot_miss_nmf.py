import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
import pickle as pkl
from sklearn.decomposition import NMF
import warnings
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, pearsonr, ttest_ind, wilcoxon
from scipy.spatial.distance import euclidean
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from PIL import Image
from collections import Counter
from tqdm import tnrange, tqdm_notebook
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import scipy as sp
from scipy.optimize import nnls
from scipy import stats
from sklearn.preprocessing import scale, MinMaxScaler
import matplotlib.colors as mcolors
import pickle
from pathlib import Path
from matplotlib.cm import get_cmap


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


def plot_scatter_plots(beta_est, beta_truth, r1, input_path, simi, mode, file, miss):
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
    fig.savefig('./results/figure/' + str(miss) + input_path + str(simi) + mode + file + str(r1) + '.png')


def rmse(y, y_pred):
    # Ensure both y and y_pred are 2D arrays with the same shape
    y = np.array(y).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)

    # Calculate RMSE
    return np.sqrt(((y - y_pred) ** 2).mean())


def factors_vs_proportions_heatmaps_real(factors, proportions, num, method, rmse_plot, input_path, file, simi, mode, r1):
    if method == "PCA":
        fc = "PC"
    if method == "SVD":
        fc = "SVD"
    if method == "ICA":
        fc = "IC"
    if method == "NMF":
        fc = "Factor"

        # Create a colormap and normalize object for correlation values
    # Set the font to Arial for all text
    plt.rcParams['font.family'] = 'Arial'
    # Define your custom colormap colors and their positions
    colors = ['purple', 'white', 'yellow']
    positions = [0.0, 0.5, 1.0]
    # Create the colormap using LinearSegmentedColormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))
    # Create a colormap and normalize object for correlation values
    # cmap = plt.get_cmap('cividis')
    cmap.set_bad((0.4, 0.4, 0.4, 0.4))  # Set alpha value to 0.4 (0 is fully transparent, 1 is fully opaque)
    norm = Normalize(vmin=-1, vmax=1)  # Set vmin and vmax to -1 and 1 for correlations
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    # Iterate over the number of missing cells

    # Define the number of rows and columns for the grid layout
    num_rows = len(proportions.columns)
    num_cols = len(factors.columns)

    if num_rows == 1:
        # Create a single subplot with two separate scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Two columns for two factors
        fig.suptitle(f'{method} on Residual: {num} Missing Cell {num} vs. Missing Cell Proportion', fontsize=16, y=0.95)
        x = list(proportions.iloc[:, 0])  # Assuming there's only one cell type
        correlations = np.zeros(2)  # Array to store correlations
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1)
        for j, factor in enumerate(factors.columns):
            y = list(factors[factor])
            ax = axes[j]  # Use the current subplot for plotting

            # Calculate Pearson's correlation coefficient
            r, p = stats.pearsonr(x, y)
            correlations[j] = r
            # Map correlation value to color
            color = scalar_map.to_rgba(r)

            # Scatter plot with color based on correlation
            ax.scatter(x, y, c='dimgrey', alpha=0.7)
            ax.set_xlabel(f'Proportions', fontsize=12)
            ax.set_ylabel(f'{fc} {factor}', fontsize=12, labelpad=0.5)
            ax.set_xlabel(f'{proportions.columns[0]} Proportions', fontsize=12)
            ax.patch.set_facecolor(color)
            ax.patch.set_alpha(1)
            # only show RMSE if relevant:
            if rmse_plot:
                # Calculate RMSE
                rmse_value = rmse(x, y)
                ax.annotate('RMSE = {:.2f}'.format(rmse_value), xy=(0.5, 0.9), xycoords='axes fraction',
                            ha='center', va='center', fontsize=10, fontweight='bold')
        # Create a colorbar for the scatter plot
        cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist(), alpha=1, pad=0.01)
        cax.set_label('Correlation (r)', fontsize=12)
        cax.set_alpha(0.4)
        plt.savefig('./results/figure/noscale' + input_path + "missing" +str(num) + str(simi) + mode + file + str(r1) + '.png')
    else:
        # Create a grid of subplots
        len_row = 6 * num - num
        len_col = 4 * num - num
        if num == 2:
            len_row = len_row + 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(len_row, len_col))
        fig.suptitle(f'{method} on Residual: {num} Missing Cells {num} vs. Missing Cells Proportion',
                     fontsize=16, y=0.93)  # Adjust the title spacing
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        # Initialize an array to store correlations
        correlations = np.zeros((num_rows, num_cols))

        # Iterate over cell types and factors
        for i, cell_type in enumerate(proportions.columns):
            for j, factor in enumerate(factors.columns):
                x = list(proportions[cell_type])
                y = list(factors[factor])
                # Use the current subplot for plotting
                ax = axes[i, j]
                # Calculate Pearson's correlation coefficient
                r, p = stats.pearsonr(x, y)
                correlations[i, j] = r

                # Map correlation value to color
                color = scalar_map.to_rgba(r)

                # Scatter plot with color based on correlation
                ax.scatter(x, y, c='dimgrey', alpha=0.7)
                ax.set_xlabel(f'{cell_type} Proportions', fontsize=12)
                ax.set_ylabel(f'{fc} {factor}', fontsize=12)
                ax.set_ylabel(f'{fc} {factor}', fontsize=12)
                ax.patch.set_facecolor(color)
                ax.patch.set_alpha(1)
                # only show RMSE if relevant:
                if rmse_plot:
                    # Calculate RMSE
                    rmse_value = rmse(x, y)
                    ax.annotate('RMSE = {:.2f}'.format(rmse_value), xy=(0.5, 0.9), xycoords='axes fraction',
                                ha='center', va='center', fontsize=10, fontweight='bold')
                    # Create a colorbar for the scatter plots
        if num > 3:
            bar_pad = 0.02
        else:
            bar_pad = 0.01
        cax = plt.colorbar(scalar_map, ax=axes.ravel().tolist(), alpha=1, pad=bar_pad)
        cax.set_label('Correlation (r)', fontsize=12)
        cax.set_alpha(1)
        plt.savefig(
            './results/figure/noscale' + input_path + "missing" + str(num) + str(simi) + mode + file + str(r1) + '.png')


def main():
    for r1 in [[12, 64, 128]]:  # [0], [128, 72, 48, 12], [128, 84, 52, 14]
        input_path = "WGBS_sim_input.txt" # "WGBS_ref_1+2.txt", "WGBS_ref_2+1.txt"
        simi = 1
        mode = "overall"
        file = "0.3False0.4"
        num_samples = 9
        num_tissues = 9
        results = []
        for miss in [0, 1, 2, 3]:
            # "0.1False0.4", "0.3False0.4", "0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"
            print(file, "miss cell type(beta)", miss)
            print("r1", r1, "\n")
            beta_truth_path = './new_results/' + str(miss) + "3000number" + input_path + str(simi) + mode + file + str(
                r1) + 'beta_true.pkl'
            # beta_truth_path = './new_results/' + str(miss) + "3000numbercfsort" + input_path + str(
            #     simi) + mode + file + 'beta_true.pkl'
            with open(beta_truth_path, "rb") as fin:
                beta_truth = pickle.load(fin)

            beta_est_path = './new_results/' + str(miss) + "3000number" + input_path + str(simi) + mode + file + str(
                r1) + 'beta_est.pkl'
            # beta_est_path = './new_results/' + str(miss) + "3000numbercfsort" + input_path + str(
            #     simi) + mode + file + 'beta_est.pkl'
            with open(beta_est_path, "rb") as fin:
                beta_est = pickle.load(fin)

            alpha_est_path = './new_results/' + str(miss) + "3000number" + input_path + str(simi) + mode + file + str(
                r1) + 'alpha_est.pkl'
            # alpha_est_path = './new_results/' + str(miss) + "3000numbercfsort" + input_path + str(
            #     simi) + mode + file + 'alpha_est.pkl'
            with open(alpha_est_path, "rb") as fin:
                alpha_est = pickle.load(fin)

            alpha_true_path = './new_results/' + str(miss) + "3000number" + input_path + str(simi) + mode + file + str(
                r1) + 'alpha_true.pkl'
            # alpha_true_path = './new_results/' + str(miss) + "3000numbercfsort" + input_path + str(
            #     simi) + mode + file + 'alpha_true.pkl'
            with open(alpha_true_path, "rb") as fin:
                alpha_true = pickle.load(fin)

            x_true_path = './new_results/' + str(miss) + "3000number" + input_path + str(simi) + mode + file + str(
                r1) + 'x_true.pkl'
            # x_true_path = './new_results/' + str(miss) + "3000numbercfsort" + input_path + str(
            #     simi) + mode + file + 'x_true.pkl'
            with open(x_true_path, "rb") as fin:
                x_true = pickle.load(fin)
            x_true = x_true.numpy()
            prop_true_path = './new_results/' + str(miss) + "3000number" + input_path + str(simi) + mode + file + str(
                r1) + 'prop_true.pkl'
            # prop_true_path = './new_results/' + str(miss) + "3000numbercfsort" + input_path + str(
            #     simi) + mode + file + 'prop_true.pkl'
            with open(prop_true_path, "rb") as fin:
                prop_true = pickle.load(fin)

            # Define colormap for coloring cell types
            cmap = get_cmap('tab20')
            # Create a dictionary to store color mapping for each cell type
            real_proportions = np.delete(prop_true.numpy(), np.s_[:miss], axis=1)
            prop_true = pd.DataFrame(real_proportions)
            cell_type_colors = {}
            estimated_proportions = alpha_est # estimated proportions of present cells.
            cell_types = prop_true.columns.tolist()
            print("Real Proportions shape:", real_proportions.shape)
            print("Estimated Proportions shape:", estimated_proportions.shape)
            # Scatter plot with colored points for each cell type
            fig, ax = plt.subplots()
            for i, cell_type in enumerate(cell_types):
                x = real_proportions[:, i]
                y = estimated_proportions[:, i]

                # Check if the cell type already has a color assigned
                if cell_type not in cell_type_colors:
                    # Assign a unique color for each cell type
                    color = cmap(len(cell_type_colors))
                    cell_type_colors[cell_type] = color
                else:
                    color = cell_type_colors[cell_type]

                ax.scatter(x, y, label=cell_type, color=color)
            m, b = np.polyfit(real_proportions.flatten(), estimated_proportions.flatten(), 1)
            plt.plot(real_proportions.flatten(), m * real_proportions.flatten() + b, color="red")
            r, p = stats.pearsonr(real_proportions.flatten(), estimated_proportions.flatten())
            rmse = np.sqrt(((x - y) ** 2).mean())
            # And annotate
            plt.annotate(f'r = {r:.2f}\nRMSE = {rmse:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontweight='bold')
            plt.xlabel('Real Proportions', fontsize=12, fontweight='bold')
            plt.ylabel('Estimated Proportions', fontsize=12, fontweight='bold')
            plt.title(f'{miss} Cell Type Missing', fontsize=12, fontweight='bold')
            plt.gca().spines['top'].set_linewidth(2)
            plt.gca().spines['right'].set_linewidth(2)
            plt.gca().spines['bottom'].set_linewidth(2)
            plt.gca().spines['left'].set_linewidth(2)

            plt.xticks(fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12, fontweight='bold')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            max_val = max([real_proportions.max(), estimated_proportions.max()])
            # ax.set_xlim(0, max_val + .03)
            # ax.set_ylim(0, max_val + .03)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.savefig('./results/figure/' + input_path + "3000missing" + str(miss) + str(simi) + mode + file + str(r1) + '.pdf')

            # l1, ccc = score(new_alpha_est, proportions.cpu().detach().numpy())
            # print("l1, ", l1, "ccc", ccc)

            # 计算整体 Pearson 相关性
            overall_alpha_corr, overall_alpha_p = pearsonr(alpha_est.flatten(), alpha_true.flatten())
            overall_beta_corr, overall_beta_p = pearsonr(beta_est.flatten(), beta_truth.flatten())

            results.append({
                'miss': miss,
                'cell_type': "overall",
                'alpha_r': overall_alpha_corr,
                'alpha_p_value': overall_alpha_p,
                'beta_r': overall_beta_corr,
                'beta_p_value': overall_beta_p
            })

            for i in range(alpha_est.shape[1]):
                alpha_corr, alpha_p_value = pearsonr(alpha_est[:, i], alpha_true[:, i])
                beta_corr, beta_p_value = pearsonr(beta_est[i, :].flatten(), beta_truth[i, :].flatten())

                results.append({
                    'miss': miss,
                    'cell_type': f'CellType_{i}',
                    'alpha_r': alpha_corr,
                    'alpha_p_value': alpha_p_value,
                    'beta_r': beta_corr,
                    'beta_p_value': beta_p_value
                })
        results_df = pd.DataFrame(results)
        output_path = './results/DTNN'+ input_path + "3000missing" + str(simi) + mode + file + 'pearson_results.xlsx'
        results_df.to_excel(output_path, index=False)


if __name__ == "__main__":
    main()

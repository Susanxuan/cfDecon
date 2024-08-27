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
import ipywidgets
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import scipy as sp
from scipy.optimize import nnls
from scipy import stats
from sklearn.preprocessing import scale, MinMaxScaler
import matplotlib.colors as mcolors
import pickle
from pathlib import Path


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
        plt.savefig('./results/figure/noscale' + input_path + "5000missing" +str(num) + str(simi) + mode + file + str(r1) + '.png')
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
            './results/figure/noscale' + input_path + "5000missing" + str(num) + str(simi) + mode + file + str(r1) + '.png')


def main():
    for r1 in [[128, 72, 48, 12]]:  # [0], [128, 72, 48, 12], [128, 84, 52, 14];
        #input_path = "WGBS_ref_1+2.txt"  # "WGBS_ref_1+2.txt", "WGBS_ref_2+1.txt"
        simi = 1
        mode = "overall"
        file = "0.3False0.4"
        num_samples = 9
        num_tissues = 9
        proportions = "./results/fractions/sample_array_50009_spar" + file + ".pkl"  #
        for input_path in ["WGBS_ref_1+2.txt", "WGBS_ref_2+1.txt"]:
            for miss in [1, 2, 3, 4, 5, 6]:
                # "0.1False0.4", "0.3False0.4", "0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"
                print(file, "miss cell type(beta)", miss)
                print("r1", r1, "\n")
                beta_truth_path = './results/' + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                    r1) + 'beta_true.pkl'
                with open(beta_truth_path, "rb") as fin:
                    beta_truth = pickle.load(fin)

                beta_est_path = './results/' + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                    r1) + 'beta_est.pkl'
                with open(beta_est_path, "rb") as fin:
                    beta_est = pickle.load(fin)

                alpha_est_path = './results/' + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                    r1) + 'alpha_est.pkl'
                with open(alpha_est_path, "rb") as fin:
                    alpha_est = pickle.load(fin)

                alpha_true_path = './results/' + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                    r1) + 'alpha_true.pkl'
                with open(alpha_true_path, "rb") as fin:
                    alpha_true = pickle.load(fin)

                x_true_path = './results/' + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                    r1) + 'x_true.pkl'
                with open(x_true_path, "rb") as fin:
                    x_true = pickle.load(fin)
                x_true = x_true.numpy()
                prop_true_path = './results/' + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                    r1) + 'prop_true.pkl'
                with open(prop_true_path, "rb") as fin:
                    prop_true = pickle.load(fin)

                x_recon_recreated = torch.matmul(torch.from_numpy(alpha_est),
                                                 torch.from_numpy(beta_est).permute(1, 0, 2))
                x_recon_np_recreated = x_recon_recreated.numpy()  # 1581 9 5
                x_recon_np_recreated = np.transpose(x_recon_np_recreated, (1, 0, 2))

                # 计算残差矩阵
                residual_matrix = x_true - x_recon_np_recreated  # 100 1581 5（现在这里是五个通道 后面就基于矩阵来做这里分通道处理）
                channels = residual_matrix.shape[2]

                ref_scale = []
                ref_scale_matrix = np.zeros((beta_est.shape[0], beta_est.shape[1], channels))
                for channel in range(channels):
                    ref_raw_val = beta_est[:, :, channel]  ##reference of bayes prism
                    clip_upper = np.quantile(ref_raw_val, 0.95)
                    ref_raw_val = np.clip(ref_raw_val, 0, clip_upper)
                    scaler = MinMaxScaler()
                    scaler.fit(ref_raw_val)
                    ref_scale.append(scaler.transform(ref_raw_val))
                    ref_scale_matrix[:, :, channel] = ref_scale[channel]

                x_recon_scale_recreated = torch.matmul(torch.from_numpy(alpha_est),
                                                       torch.from_numpy(ref_scale_matrix).float().permute(1, 0, 2))
                x_recon_scale_np_recreated = x_recon_scale_recreated.numpy()  # 1581 9 5
                x_recon_scale_np_recreated = np.transpose(x_recon_scale_np_recreated, (1, 0, 2))
                residuals_scaled = x_true - x_recon_scale_np_recreated

                # 按通道处理残差矩阵
                residuals_shift = []
                residuals_shift_noscale = []

                # 处理每个通道
                for channel in range(channels):
                    residuals_channel_scaled = residuals_scaled[:, :, channel]

                    min_val = abs(np.min(residuals_channel_scaled))
                    residuals_channel_shift = residuals_channel_scaled + min_val
                    residuals_shift.append(residuals_channel_shift)

                    residuals_channel = residual_matrix[:, :, channel]

                    min_val = abs(np.min(residuals_channel))
                    residuals_channel = residuals_channel + min_val
                    residuals_shift_noscale.append(residuals_channel)

                # 转换为numpy数组
                residuals_scaled = np.array(residuals_scaled)
                residuals_shift = np.transpose(np.array(residuals_shift), (1, 2, 0))
                residuals_shift_noscale = np.transpose(np.array(residuals_shift_noscale), (1, 2, 0))

                missing_cell_prop = pd.DataFrame(prop_true[:, :miss])

                # # 依据通道
                # res_nmf = []
                # for channel in range(channels):
                #     channel_data = residuals_shift[:, :, channel]
                #     nmf = NMF(n_components=miss+1, max_iter=15000, init="nndsvd")
                #     res_nmf_channel = nmf.fit_transform(channel_data)
                #     # res_nmf_df = pd.DataFrame(res_nmf_channel,
                #     #                           columns=[f'NMF_{i + 1}_Channel_{channel + 1}' for i in range(miss+1)])
                #     pseudo_raw_val = res_nmf_channel
                #     scaler = MinMaxScaler()
                #     scaler.fit(pseudo_raw_val)
                #     pseudo_scale = scaler.transform(pseudo_raw_val)
                #     res_nmf_df = pd.DataFrame(pseudo_scale)
                #
                #     res_nmf.append(res_nmf_df)
                # res_nmf_array = np.array([df.values for df in res_nmf])
                # res_nmf_avg = np.mean(res_nmf_array, axis=0)
                # res_nmf_avg = pd.DataFrame(res_nmf_avg, columns=res_nmf[0].columns)

                # 将三维矩阵转换为二维矩阵
                samples, features, channels = residuals_shift.shape
                residuals_2d = residuals_shift_noscale.reshape(samples, -1)  # (samples, features * channels)
                num_nmf_components = miss + 1  # 假设miss已经定义
                nmf = NMF(n_components=num_nmf_components, max_iter=60000, init="nndsvd")
                res_nmf_2d = nmf.fit_transform(residuals_2d)
                # res_nmf_avg = pd.DataFrame(res_nmf_2d, columns=[f'NMF_{i + 1}' for i in range(num_nmf_components)])
                scaler = MinMaxScaler()
                res_nmf_scaled = scaler.fit_transform(res_nmf_2d)
                res_nmf_avg = pd.DataFrame(res_nmf_scaled, columns=[f'NMF_{i + 1}' for i in range(num_nmf_components)])

                rmse_plot = True
                factors_vs_proportions_heatmaps_real(res_nmf_avg, missing_cell_prop, miss, "NMF", rmse_plot,
                                                     input_path, file, simi, mode, r1)

                # 反推出缺失细胞类型的比例
                # l1, ccc = score(new_alpha_est, proportions.cpu().detach().numpy())
                # print("l1, ", l1, "ccc", ccc)



if __name__ == "__main__":
    main()

import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from misc import *
from sklearn.model_selection import train_test_split


def extract_values(file_name):
    """Extract sparse, ifrare, and rare values from the filename."""
    if "True" in file_name:
        sparse, rest = file_name.split("True")
        ifrare = "True"
    else:
        sparse, rest = file_name.split("False")
        ifrare = "False"
    rare = rest
    return float(sparse), ifrare, float(rare)


def train_model(data_df, num_samples, num_cell, proportions, simi, sparse, ifrare, rare, output_directory, mode,
                adaptive, r1, r2,
                alpha, beta, use_norm):
    """Train the autoencoder model."""
    props = pkl.load(open(proportions, 'rb'))
    np.random.seed(1)
    if simi == 2:
        simulated_x_1, true_beta_1, simulated_x_2, true_beta_2 = simulate_arrays(data_df, int(num_samples),
                                                                                 int(num_cell),
                                                                                 proportions=props, simi=simi,
                                                                                 sparse=sparse, ifrare=ifrare,
                                                                                 rare=rare)
        data_1 = torch.tensor(simulated_x_1)
        label_1 = torch.tensor(props[:int(props.shape[0] / 2), :])
        data_2 = torch.tensor(simulated_x_2)
        label_2 = torch.tensor(props[int(props.shape[0] / 2):, :])
        train_x_1, test_x_1, train_y_1, test_y_1 = train_test_split(data_1, label_1, test_size=0.1, random_state=0)
        train_x_2, test_x_2, train_y_2, test_y_2 = train_test_split(data_2, label_2, test_size=0.1, random_state=0)
        train_x = torch.cat((train_x_1, train_x_2), dim=0)
        train_y = torch.cat((train_y_1, train_y_2), dim=0)
        test_x = torch.cat((test_x_1, test_x_2), dim=0)
        test_y = torch.cat((test_y_1, test_y_2), dim=0)
        true_beta = None
        with open(output_directory + "beta_true_1.pkl", "wb") as f:
            pkl.dump(true_beta_1, f)
        with open(output_directory + "beta_true_2.pkl", "wb") as f:
            pkl.dump(true_beta_2, f)
    else:
        x, true_beta = simulate_arrays(data_df, int(num_samples), int(num_cell), proportions=props,
                                       simi=simi, sparse=sparse, ifrare=ifrare, rare=rare)
        # if miss case : simulate_arrays_miss(data_df, int(num_samples), int(num_cell), simi=simi, miss=miss)
        data = torch.tensor(x)
        label = torch.tensor(props)
        min_val_data = torch.min(data)
        max_val_data = torch.max(data)
        normalized_data = (data - min_val_data) / (max_val_data - min_val_data)
        train_x, test_x, train_y, test_y = train_test_split(normalized_data, label, test_size=0.1, random_state=0)
    l1, ccc, pred, sign = test_AE_function(train_x, train_y, test_x, test_y, outdir=output_directory, adaptive=adaptive,
                                           mode=mode, r1=r1, r2=r2, alpha=alpha, beta=beta, use_norm=use_norm)
    print('L1:', l1, 'CCC:', ccc)
    with open(output_directory + "alpha_est.pkl", "wb") as f:
        pkl.dump(pred, f)
    with open(output_directory + "alpha_true.pkl", "wb") as f:
        pkl.dump(test_y, f)
    with open(output_directory + "beta_est.pkl", "wb") as f:
        pkl.dump(sign, f)
    with open(output_directory + "beta_true.pkl", "wb") as f:
        pkl.dump(true_beta, f)


def test_model(data_df, num_samples, num_cell, n, fix, output_directory, mode, adaptive, r1, r2,
               alpha, beta, use_norm):
    """Test the autoencoder model."""
    x, true_beta, test_samples, training_props = simulate_arrays_new(data_df, int(num_samples), n=n, c=num_cell,
                                                                     fix_c=fix, UXM=True)
    data = torch.tensor(x)
    min_val_data = torch.min(data)
    max_val_data = torch.max(data)
    normalized_data = (data - min_val_data) / (max_val_data - min_val_data)
    label = torch.tensor(training_props)
    min_val = np.min(test_samples)
    max_val = np.max(test_samples)
    normalized_samples = (test_samples - min_val) / (max_val - min_val)
    pred, sign = predict_AE_function(normalized_data, label, normalized_samples,
                                     int(num_cell),
                                     outdir=output_directory, adaptive=adaptive, mode=mode,
                                     r1=r1, r2=r2, alpha=alpha, beta=beta, use_norm=use_norm,
                                     true_beta=true_beta)
    with open(output_directory + "beta_est.pkl", "wb") as f: pkl.dump(sign, f)
    with open(output_directory + "beta_true.pkl", "wb") as f: pkl.dump(true_beta, f)
    with open(output_directory + "alpha_est.pkl", "wb") as f: pkl.dump(pred, f)


def main():
    parser = argparse.ArgumentParser(description="Autoencoder model for WGBS reference data.")
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory path')
    parser.add_argument('--input_path', type=str, required=True, help='Input path for data file')
    parser.add_argument('--mode', type=str, default="overall", help='Training mode (overall/high-resolution)')
    parser.add_argument('--train', type=int, default=1, help='Set to 1 for training')
    parser.add_argument('--test', type=int, default=0, help='Set to 1 for testing')
    parser.add_argument('--use_norm', type=str, default="False", help='Set normalization (True/False)')
    parser.add_argument('--alpha', type=int, default=10, help='Alpha value for training')
    parser.add_argument('--beta', type=int, default=1, help='Beta value for training')
    parser.add_argument('--simi', type=int, default=1, help='Set similarity type')
    parser.add_argument('--num_samples', type=int, default=9, help='Set silico/cfdna sample')
    parser.add_argument('--num_cell', type=int, default=9, help='Set the number of cell type')
    parser.add_argument('--adaptive', type=bool, default=True, help='if need refinement stage')
    parser.add_argument('--proportions', type=str,
                        default="./results/fractions/sample_array_10009_spar0.1False0.4.pkl",
                        help='the training predefined proportions')
    parser.add_argument('--r1', nargs='+', type=int, help='the parameters of encoder')  # [128, 72, 48, 12]
    parser.add_argument('--r2', nargs='+', type=int, help='the parameters of decoder')  # [12, 64, 128]
    parser.add_argument('--file', type=str, default="0.1False0.4", help='the name of file')
    parser.add_argument('--n', type=int, default=5000, help='the training samples')
    parser.add_argument('--fix', type=str, default="True", help='the mode for UXM')
    args = parser.parse_args()
    data_df = pd.read_csv(args.input_path, delimiter="\t", header=None, skiprows=1)
    sparse, ifrare, rare = extract_values(args.file)
    if args.train:
        train_model(data_df, args.num_samples, args.num_cell, args.proportions, args.simi, sparse, ifrare, rare,
                    args.output_dir,
                    args.mode,
                    args.adaptive, args.r1, args.r2, args.alpha, args.beta, args.use_norm)
    if args.test:
        test_model(data_df, args.num_samples, args.num_cell, args.n, args.fix, args.output_dir, args.mode,
                   args.adaptive, args.r1, args.r2, args.alpha, args.beta, args.use_norm)


if __name__ == "__main__":
    main()

# python ./train/main.py --output_dir ./results/ --input_path ./data/WGBS_ref_1+2.txt --train 1 --r1 128 72 48 12 --r2 12 64 128

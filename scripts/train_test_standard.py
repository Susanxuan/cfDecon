import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from Multi_channel_AE_celfeer import *
from sklearn.model_selection import KFold
import argparse
import pickle as pkl
from sklearn.model_selection import train_test_split


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run deconvolution using Multi-channel Autoencoder")

    # 添加命令行参数
    parser.add_argument("--input_path", type=str, default="./data/ALS.txt", help="Path to the input file")
    parser.add_argument("--output_directory", type=str, default="./results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples")
    parser.add_argument("--num_tissues", type=int, default=19, help="Number of cell types")
    parser.add_argument("--unknowns", type=int, default=0, help="Number of unknown samples")
    parser.add_argument("--parallel_job_id", type=int, default=1, help="Parallel job ID")
    parser.add_argument("--simi", type=int, default=1, help="Simi parameter, set as 1 for real data")
    parser.add_argument("--use_norm", action="store_true", help="Use LayerNorm or not")
    parser.add_argument("--mode", type=str, default="overall", help="Mode: overall or high-resolution")
    parser.add_argument("--train", type=int, default=1, help="Set to 1 if proportions are available")
    parser.add_argument("--test", type=int, default=1, help="Set to 0 if proportions are not available")
    parser.add_argument("--file", type=str, default="0.1False0.4", help="File identifier for proportions")
    parser.add_argument("--r1", type=int, nargs='+', default=[128, 72, 48, 12], help="List of values for r1")
    parser.add_argument("--r2", type=int, nargs='+', default=[12, 48, 72, 128], help="List of values for r2")

    return parser.parse_args()


def main():
    # 使用 argparse 解析命令行参数
    args = parse_arguments()

    # 提取参数
    input_path = args.input_path
    output_directory = args.output_directory
    num_samples = args.num_samples
    num_tissues = args.num_tissues
    unknowns = args.unknowns
    parallel_job_id = args.parallel_job_id
    simi = args.simi
    use_norm = args.use_norm
    mode = args.mode
    train = args.train
    test = args.test
    file = args.file
    r1 = args.r1
    r2 = args.r2

    # 主体代码
    proportions = f"./results/fractions/sample_array_1000{num_tissues}_spar{file}.pkl"
    sparse, ifrare, rare = extract_values(file)
    adaptive = True
    np.random.seed(parallel_job_id)
    if not os.path.exists(output_directory) and parallel_job_id == 1:
        os.makedirs(output_directory)
        print("made " + output_directory + "/")
        print()
    else:
        print("writing to " + output_directory + "/")

    data_df = pd.read_csv(input_path, delimiter="\t", header=None, skiprows=1)

    print(f"finished reading {input_path}", "file", file, "mode", mode, "adaptive", adaptive)
    print()
    print(f"beginning generation of {output_directory}")
    print()
    alpha = 20
    beta = 1

    print("r1, r2", r1, r2, "\n")
    if train == 1:
        props = pkl.load(open(proportions, 'rb'))
        if simi == 2:
            # simulated 2 data
            simulated_x_1, true_beta_1, simulated_x_2, true_beta_2 = simulate_arrays(data_df,
                                                                                     int(num_samples),
                                                                                     int(unknowns),
                                                                                     proportions=props,
                                                                                     simi=simi,
                                                                                     sparse=sparse,
                                                                                     ifrare=ifrare,
                                                                                     rare=rare)
            data_1 = torch.tensor(simulated_x_1)
            label_1 = torch.tensor(props[:int(props.shape[0] / 2), :])
            data_2 = torch.tensor(simulated_x_2)
            label_2 = torch.tensor(props[int(props.shape[0] / 2):, :])
            train_x_1, test_x_1, train_y_1, test_y_1 = train_test_split(data_1, label_1, test_size=0.1,
                                                                        random_state=0)
            train_x_2, test_x_2, train_y_2, test_y_2 = train_test_split(data_2, label_2, test_size=0.1,
                                                                        random_state=0)

            train_x = torch.cat((train_x_1, train_x_2), dim=0)
            train_y = torch.cat((train_y_1, train_y_2), dim=0)

            # # Generate random permutation of indices
            # random_indices = torch.randperm(train_x.size(0))

            # # Use indexing to reorder the tensor
            # train_x = train_x[random_indices]
            # train_y = train_y[random_indices]

            test_x = torch.cat((test_x_1, test_x_2), dim=0)
            test_y = torch.cat((test_y_1, test_y_2), dim=0)
        else:
            x, true_beta = simulate_arrays(data_df, int(num_samples), int(unknowns), proportions=props,
                                           simi=simi, sparse=sparse, ifrare=ifrare, rare=rare)
            data = torch.tensor(x)
            label = torch.tensor(props)
            train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.1,
                                                                random_state=0)
        L1 = []
        CCC = []
        if adaptive:
            l1, ccc, pred, sign = test_AE_function(train_x, train_y, test_x, test_y,
                                                   outdir=output_directory,
                                                   adaptive=adaptive, mode=mode, r1=r1, r2=r2,
                                                   alpha=alpha,
                                                   beta=beta, use_norm=use_norm)
            print('l1, ccc:', l1, ccc)
            L1.append(l1)
            CCC.append(ccc)
            with open(output_directory + "/" + input_path + str(simi) + mode + file + str(
                    r1) + str(alpha) + str(beta) + "alpha_est.pkl",
                      "wb") as f:  # "/COAD" +
                pkl.dump(pred, f)
            with open(
                    output_directory + "/" + input_path + str(simi) + mode + file + str(
                        r1) + str(alpha) + str(beta) + "alpha_true.pkl",
                    "wb") as f:
                pkl.dump(props, f)
            with open(output_directory + "/" + input_path + str(simi) + mode + file + str(
                    r1) + str(alpha) + str(beta) + "beta_est.pkl",
                      "wb") as f:
                pkl.dump(sign, f)

            if simi == 2:
                with open(
                        output_directory + "/" + input_path + str(simi) + mode + file + str(
                            r1) + "beta_true_1.pkl",
                        "wb") as f:
                    pkl.dump(true_beta_1, f)
                with open(
                        output_directory + "/" + input_path + str(simi) + mode + file + str(
                            r1) + "beta_true_2.pkl",
                        "wb") as f:
                    pkl.dump(true_beta_2, f)
            else:
                with open(output_directory + "/new" + input_path + str(simi) + mode + file + str(
                        r1) + "beta_true.pkl",
                          "wb") as f:
                    pkl.dump(true_beta, f)
        else:
            l1, ccc, pred, sign = test_AE_function(train_x, train_y, test_x, test_y,
                                                   outdir=output_directory,
                                                   adaptive=adaptive, mode=mode, r1=r1, r2=r2,
                                                   alpha=alpha,
                                                   beta=beta, use_norm=use_norm)
            print('l1, ccc:', l1, ccc)
            L1.append(l1)
            CCC.append(ccc)
            with open(output_directory + "/nonada" + input_path + str(simi) + mode + file + str(
                    r1) + str(alpha) + str(beta) + "alpha_est.pkl", "wb") as f:
                pkl.dump(pred, f)
            with open(output_directory + "/nonada" + input_path + str(simi) + mode + file + str(
                    r1) + str(alpha) + str(beta) + "alpha_true.pkl", "wb") as f:
                pkl.dump(props, f)
            with open(output_directory + "/nonada" + input_path + str(simi) + mode + file + str(
                    r1) + str(alpha) + str(beta) + "beta_est.pkl", "wb") as f:
                pkl.dump(sign, f)

    if test == 1:
        x, true_beta, test_samples, training_props = simulate_arrays(data_df, int(num_samples),
                                                                     int(unknowns), sparse=sparse,
                                                                     ifrare=ifrare, rare=rare)
        data = torch.tensor(x)
        label = torch.tensor(training_props)

        if adaptive:
            pred, sigm = predict_AE_function(data, label, test_samples, int(num_tissues),
                                             outdir=output_directory, adaptive=adaptive, mode=mode,
                                             r1=r1,
                                             r2=r2, alpha=alpha, beta=beta, use_norm=use_norm)
            with open(output_directory + "/test" + input_path + str(simi) + mode + file + str(
                    r1) + str(alpha) + str(beta) + "alpha_est.pkl",
                      "wb") as f:
                pkl.dump(pred, f)
            with open(output_directory + "/test" + input_path + str(simi) + mode + file + str(
                    r1) + str(alpha) + str(beta) + "beta_est.pkl", "wb") as f:
                pkl.dump(sigm, f)
            with open(output_directory + "/test" + input_path + str(simi) + mode + file + str(
                    r1) + str(alpha) + str(beta) + "beta_true.pkl", "wb") as f:  # COAD_
                pkl.dump(true_beta, f)
            data_df_header = pd.read_csv(
                input_path, delimiter="\t", nrows=1
            )
            # get header for output files
            samples, tissues = get_header(data_df_header, num_samples, unknowns)

            # write estimates as text files
            # output_alpha_file=output_directory + "/alpha_est.csv"
            # write_output(output_alpha_file, pred, tissues, samples, unknowns)
        else:
            pred, sign = predict_AE_function(data, label, test_samples, int(num_tissues),
                                             outdir=output_directory,
                                             adaptive=adaptive, mode=mode, r1=r1, r2=r2, alpha=alpha,
                                             beta=beta, use_norm=use_norm)
            # print(pred)
            # pred.to_csv(output_directory + "/alpha_est.csv")
            with open(output_directory + "/nonadatest" + input_path + str(simi) + mode + file + str(
                    r1) + str(alpha) + str(beta) + "alpha_est.pkl", "wb") as f:
                pkl.dump(pred, f)
            with open(output_directory + "/nonadatest" + input_path + str(simi) + mode + file + str(
                    r1) + str(alpha) + str(beta) + "beta_est.pkl", "wb") as f:
                pkl.dump(sign, f)
            with open(output_directory + "/nonadatest" + input_path + str(simi) + mode + file + str(
                    r1) + str(alpha) + str(beta) + "beta_true.pkl", "wb") as f:  # COAD_
                pkl.dump(true_beta, f)

            data_df_header = pd.read_csv(
                input_path, delimiter="\t", nrows=1
            )
            # get header for output files
            samples, tissues = get_header(data_df_header, num_samples, unknowns)


if __name__ == "__main__":
    main()

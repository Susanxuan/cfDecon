import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from Multi_channel_AE_celfeer import *
# from sklearn.model_selection import KFold


def extract_values(file_name):
    # Split the filename based on the boolean value (True/False)
    if "True" in file_name:
        sparse, rest = file_name.split("True")
        ifrare = "True"
    else:
        sparse, rest = file_name.split("False")
        ifrare = "False"
    rare = rest
    return float(sparse), ifrare, float(rare)


def main():
    # 直接在代码中指定参数
    input_path = "./data/UXM_ref31_100m.bed"  # WGBS_ref_2+1.txt
    output_directory = "./results"
    num_samples = 1  # 52
    unknowns = 0
    parallel_job_id = 1
    simi = 1  # 这里
    train = 1
    test = 1
    use_norm = True
    for mode in ["overall"]:
        for fix in ["False"]:  # "WGBS_ref_1+2.txt", "WGBS_ref_2+1.txt" "UXM_39.bed"
            for n in [1000, 5000, 7500, 10000]:  # "0.1False0.4", "0.3False0.4",  32 64 128
                proportions = "./results/fractions/UXM_sample_array_" + str(n) + "31" + fix + ".pkl"  #
                sparse, ifrare, rare = None, None, None
                #  "0.1False0.4", "0.3False0.4","0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"
                adaptive = True
                # mode = "overall"  # "high-resolution"  # high-resolution(dictionary  overall
                # adaptive = True
                np.random.seed(parallel_job_id)
                num_tissues = 31
                test = 1  # WGBS不用这个这个就可以直接设置为0 这个就是没有proportions
                if not os.path.exists(output_directory) and parallel_job_id == 1:
                    os.makedirs(output_directory)
                    print("made " + output_directory + "/")
                    print()
                else:
                    print("writing to " + output_directory + "/")

                # data_df = pd.read_csv(input_path, header=None, delimiter="\t")

                data_df = pd.read_csv(input_path, delimiter="\t", header=None,
                                      skiprows=1)  # read input samples/reference data

                # columns_to_drop = list(range(61, data_df.shape[1]))
                # data_df = data_df.drop(columns=columns_to_drop)

                # 删除指定的列（263, 264, 265）
                # columns_to_drop = [263, 264, 265]
                # data_df = data_df.drop(columns=columns_to_drop)

                print(f"finished reading {input_path}", n, fix, "mode", mode, "adaptive", adaptive)
                print()

                print(f"beginning generation of {output_directory}")
                print()
                alpha = 20
                beta = 1
                for r1, r2 in [([128, 72, 48, 12], [36, 128])]:# ([128, 72, 48, 12], [36, 128])
                    # ([0], [0]) ([128, 72, 48, 12], [12, 48, 72, 128]), ([128, 84, 52, 14], [12, 48, 72, 108])
                    # ([128, 72, 48, 12], [12, 64, 128])  ([128, 84, 52, 14], [12, 64, 128]), ([128, 64, 18], [18, 64, 128])
                    # ([128, 72, 48, 12], [32, 128]), ([156, 96, 48, 16], [36, 156])
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
                            with open(output_directory + "/" + input_path + str(simi) + mode + str(
                                    r1) + str(n) + fix + str(alpha) + str(beta) + "alpha_est.pkl",
                                      "wb") as f:  # "/COAD" +
                                pkl.dump(pred, f)
                            with open(
                                    output_directory + "/" + input_path + str(simi) + mode + str(
                                        r1) + str(n) + fix + str(alpha) + str(beta) + "alpha_true.pkl",
                                    "wb") as f:
                                pkl.dump(props, f)
                            with open(output_directory + "/" + input_path + str(simi) + mode + str(
                                    r1) + str(n) + fix + str(alpha) + str(beta) + "beta_est.pkl",
                                      "wb") as f:
                                pkl.dump(sign, f)

                            if simi == 2:
                                with open(
                                        output_directory + "/" + input_path + str(simi) + mode + str(
                                            r1) + "beta_true_1.pkl",
                                        "wb") as f:
                                    pkl.dump(true_beta_1, f)
                                with open(
                                        output_directory + "/" + input_path + str(simi) + mode + str(
                                            r1) + "beta_true_2.pkl",
                                        "wb") as f:
                                    pkl.dump(true_beta_2, f)
                            else:
                                with open(output_directory + "/new" + input_path + str(simi) + mode + str(
                                        r1) + str(n) + fix + "beta_true.pkl",
                                          "wb") as f:
                                    pkl.dump(true_beta, f)
                        else:
                            l1, ccc, pred, sign = test_AE_function(train_x, train_y, test_x, test_y,
                                                                   outdir=output_directory,
                                                                   adaptive=adaptive, mode=mode, r1=r1, r2=r2,
                                                                   alpha=alpha, beta=beta, use_norm=use_norm)
                            print('l1, ccc:', l1, ccc)
                            L1.append(l1)
                            CCC.append(ccc)
                            with open(output_directory + "/nonada" + input_path + str(simi) + mode + str(
                                    r1) + str(alpha) + str(beta) + "alpha_est.pkl", "wb") as f:
                                pkl.dump(pred, f)
                            with open(output_directory + "/nonada" + input_path + str(simi) + mode + str(
                                    r1) + str(alpha) + str(beta) + "alpha_true.pkl", "wb") as f:
                                pkl.dump(props, f)
                            with open(output_directory + "/nonada" + input_path + str(simi) + mode + str(
                                    r1) + str(alpha) + str(beta) + "beta_est.pkl", "wb") as f:
                                pkl.dump(sign, f)
                    if test == 1:
                        x, true_beta, test_samples, training_props = simulate_arrays_new(data_df, int(num_samples),
                                                                                     int(unknowns), n = n,
                                                                                     fix_c= fix )
                        data = torch.tensor(x)
                        label = torch.tensor(training_props)

                        if adaptive:
                            pred, sigm = predict_AE_function(data, label, test_samples, int(num_tissues),
                                                             outdir=output_directory, adaptive=adaptive, mode=mode,
                                                             r1=r1, r2=r2, alpha=alpha, beta=beta, use_norm=use_norm)
                            with open(output_directory + "/test" + input_path + str(simi) + mode + str(
                                    r1) + str(n) + fix +str(alpha) + str(beta) + "alpha_est.pkl",
                                      "wb") as f:
                                pkl.dump(pred, f)
                            with open(output_directory + "/test" + input_path + str(simi) + mode + str(
                                    r1) + str(n) + fix + str(alpha) + str(beta) + "beta_est.pkl", "wb") as f:
                                pkl.dump(sigm, f)
                            with open(output_directory + "/test" + input_path + str(simi) + mode + str(
                                    r1) + str(n) + fix + str(alpha) + str(beta) + "beta_true.pkl", "wb") as f:  # COAD_
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
                                                             beta=beta,use_norm=use_norm)
                            # print(pred)
                            # pred.to_csv(output_directory + "/alpha_est.csv")
                            with open(output_directory + "/nonadatest" + input_path + str(simi) + mode + str(
                                    r1) + str(alpha) + str(beta) + "alpha_est.pkl", "wb") as f:
                                pkl.dump(pred, f)
                            with open(output_directory + "/nonadatest" + input_path + str(simi) + mode  + str(
                                    r1) + str(alpha) + str(beta) + "beta_est.pkl", "wb") as f:
                                pkl.dump(sign, f)
                            with open(output_directory + "/nonadatest" + input_path + str(simi) + mode + str(
                                    r1) + str(alpha) + str(beta) + "beta_true.pkl", "wb") as f:  # COAD_
                                pkl.dump(true_beta, f)

                            data_df_header = pd.read_csv(
                                input_path, delimiter="\t", nrows=1
                            )
                            # get header for output files
                            samples, tissues = get_header(data_df_header, num_samples, unknowns)


if __name__ == "__main__":
    main()

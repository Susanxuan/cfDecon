import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from Multi_channel_AE_celfeer import *
from sklearn.model_selection import KFold


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
    # input_path = "WGBS_sim_input.txt"  # WGBS_ref_2+1.txt
    output_directory = "./results"
    num_samples = 9  # 52
    unknowns = 0
    parallel_job_id = 1
    num_tissues = 9
    file = "0.3False0.4"
    simi = 1
    use_norm = False
    mode = "overall"  # "high-resolution"  # high-resolution(dictionary  overall
    adaptive = True
    sparse, ifrare, rare = extract_values(file)
    props = generate_sample_array(5000, 9, sparse=True, sparse_prob=sparse,
                                           rare=(ifrare == str(True)), rare_percentage=rare)
    test = 0
    train = 1
    for input_path in ["./data/WGBS_ref_2+1.txt", "./data/WGBS_ref_1+2.txt"]:  # "WGBS_ref_1+2.txt", "WGBS_ref_2+1.txt"
        for miss in [0, 1, 2, 3, 4, 5, 6]:  # "0.1False0.4", "0.3False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"
            # , "0.3False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3" "0.3True0.1", "0.3True0.2", "0.3True0.3"
            proportions = "./results/fractions/sample_array_50009_spar" + file + ".pkl"  #
            np.random.seed(parallel_job_id)
            unknowns_list = [int(x) for x in unknowns.split(",")] if unknowns else None
            # make output directory if it does not exist
            if not os.path.exists(output_directory) and parallel_job_id == 1:
                os.makedirs(output_directory)
                print("made " + output_directory + "/")
                print()
            else:
                print("writing to " + output_directory + "/")

            # data_df = pd.read_csv(input_path, header=None, delimiter="\t")

            data_df = pd.read_csv(input_path, delimiter="\t", header=None,
                                  skiprows=1)  # read input samples/reference data

            # 删除指定的列（263, 264, 265）
            # columns_to_drop = [263, 264, 265]
            # data_df = data_df.drop(columns=columns_to_drop)

            print(f"finished reading {input_path}", "file", file, "number: missing cell type(beta)", miss)
            print()

            print(f"beginning generation of {output_directory}")
            print()

            for r1, r2 in [([128, 72, 48, 12], [12, 48, 72, 128])]:  #([128, 72, 48, 12], [12, 64, 128])
                # ([0], [0]) ([128, 72, 48, 12], [12, 48, 72, 128]), ([128, 84, 52, 14 ], [12, 48, 72, 108])
                # ([128, 72, 48, 12], [12, 64, 128])
                # Load file containing proportions
                print("r1, r2", r1, r2, "\n")
                if train == 1:
                    props = pkl.load(open(proportions, 'rb'))
                    # missing_props = np.delete(props, miss, axis=1)  # np.delete(props, miss, axis=1)
                    missing_props = np.delete(props, np.s_[:miss], axis=1)
                    if simi == 2:
                        # simulated 2 data
                        simulated_x_1, true_beta_1, simulated_x_2, true_beta_2 = simulate_arrays_miss(data_df,
                                                                                                 int(num_samples),
                                                                                                 int(unknowns),
                                                                                                 proportions=props,
                                                                                                 simi=simi,
                                                                                                 sparse=sparse,
                                                                                                 ifrare=ifrare,
                                                                                                 rare=rare, miss=miss)
                        data_1 = torch.tensor(simulated_x_1)
                        label_1 = torch.tensor(missing_props[:int(missing_props.shape[0] / 2), :])
                        data_2 = torch.tensor(simulated_x_2)
                        label_2 = torch.tensor(missing_props[int(missing_props.shape[0] / 2):, :])
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
                        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
                    else:
                        x, true_beta = simulate_arrays_miss(data_df, int(num_samples), int(unknowns), proportions=props,
                                                       simi=simi, sparse=sparse, ifrare=ifrare, rare=rare, miss=miss)
                        data = torch.tensor(x)
                        label = torch.tensor(props)
                        train_x, test_x, train_y_ori, test_y_ori = train_test_split(data, label, test_size=0.1, random_state=0)
                        train_y = np.delete(train_y_ori, np.s_[:miss], axis=1)
                        test_y = np.delete(test_y_ori, np.s_[:miss], axis=1)
                        with open(output_directory + "/" + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                                r1) + "x_true.pkl", "wb") as f:
                            pkl.dump(test_x, f)
                        with open(output_directory + "/" + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                                r1) + "prop_true.pkl", "wb") as f:
                            pkl.dump(test_y_ori, f)
                    L1 = []
                    CCC = []
                    if adaptive:
                        l1, ccc, pred, sign = test_AE_function(train_x, train_y, test_x, test_y,
                                                               outdir=output_directory,
                                                               adaptive=adaptive, mode=mode, r1=r1, r2=r2, use_norm=use_norm)
                        print('l1, ccc:', l1, ccc)
                        L1.append(l1)
                        CCC.append(ccc)
                        with open(output_directory + "/" + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                                r1) + "alpha_est.pkl",
                                  "wb") as f:  # "/COAD" +
                            pkl.dump(pred, f)
                        with open(
                                output_directory + "/" + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                                    r1) + "alpha_true.pkl",
                                "wb") as f:
                            pkl.dump(test_y, f)
                        with open(output_directory + "/" + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                                r1) + "beta_est.pkl",
                                  "wb") as f:
                            pkl.dump(sign, f)

                        if simi == 2:
                            with open(
                                    output_directory + "/" + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                                        r1) + "beta_true_1.pkl",
                                    "wb") as f:
                                pkl.dump(true_beta_1, f)
                            with open(
                                    output_directory + "/" + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                                        r1) + "beta_true_2.pkl",
                                    "wb") as f:
                                pkl.dump(true_beta_2, f)
                        else:
                            with open(output_directory + "/" + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                                    r1) + "beta_true.pkl",
                                      "wb") as f:
                                pkl.dump(true_beta, f)
                    else:
                        l1, ccc, pred, sign = test_AE_function(train_x, train_y, test_x, test_y,
                                                               outdir=output_directory,
                                                               adaptive=adaptive, mode=mode, r1=r1, r2=r2, use_norm=use_norm)
                        print('l1, ccc:', l1, ccc)
                        L1.append(l1)
                        CCC.append(ccc)
                        with open(output_directory + "/nonada" + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                                r1) + "alpha_est.pkl", "wb") as f:
                            pkl.dump(pred, f)
                        with open(output_directory + "/nonada" + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                                r1) + "alpha_true.pkl", "wb") as f:
                            pkl.dump(missing_props, f)
                        with open(output_directory + "/nonada" + str(miss) + "5000number" + input_path + str(simi) + mode + file + str(
                                r1) + "beta_est.pkl", "wb") as f:
                            pkl.dump(sign, f)
                if test == 1:
                    # training_props = np.delete(props, miss, axis=1) # np.delete(props, miss, axis=1)
                    testing_props = np.delete(props, np.s_[:miss], axis=1)
                    x, true_beta, test_samples, testing_props = simulate_arrays_miss(data_df, int(num_samples),
                                                                                 int(unknowns), sparse=sparse,
                                                                                 ifrare=ifrare, rare=rare, miss=miss,
                                                                                      training_props=testing_props)
                    with open(output_directory + "/test" + input_path + str(simi) + mode + file + str(
                            r1) + "x_recon.pkl", "wb") as f:
                        pkl.dump(test_samples, f)
                    data = torch.tensor(x)
                    label = torch.tensor(testing_props)

                    if adaptive:
                        pred, sigm = predict_AE_function(data, label, test_samples, int(num_tissues),
                                                         outdir=output_directory, adaptive=adaptive, mode=mode, r1=r1,
                                                         r2=r2,use_norm=use_norm)
                        with open(output_directory + "/test" + str(miss)+ "5000number" +input_path + str(simi) + mode + file + str(
                                r1) + "alpha_est.pkl", "wb") as f:
                            pkl.dump(pred, f)
                        with open(output_directory + "/test" + str(miss) + "5000number" +input_path + str(simi) + mode + file + str(
                                r1) + "beta_est.pkl", "wb") as f:
                            pkl.dump(sigm, f)
                        with open(output_directory + "/test" + str(miss) + "5000number" +input_path + str(simi) + mode + file + str(
                                r1) + "beta_true.pkl", "wb") as f:  # COAD_
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
                                                         adaptive=adaptive, mode=mode, r1=r1, r2=r2,use_norm=use_norm)
                        # print(pred)
                        # pred.to_csv(output_directory + "/alpha_est.csv")
                        with open(output_directory + "/nonadatest" + str(miss) + "5000number" +input_path + str(simi) + mode + file + str(
                                r1) + "alpha_est.pkl", "wb") as f:
                            pkl.dump(pred, f)
                        with open(output_directory + "/nonadatest" + str(miss) + "5000number" +input_path + str(simi) + mode + file + str(
                                r1) + "beta_est.pkl", "wb") as f:
                            pkl.dump(sign, f)
                        with open(output_directory + "/nonadatest" + str(miss) + "5000number" +input_path + str(simi) + mode + file + str(
                                r1) + "beta_true.pkl", "wb") as f:  # COAD_
                            pkl.dump(true_beta, f)

                        data_df_header = pd.read_csv(
                            input_path, delimiter="\t", nrows=1
                        )
                        # get header for output files
                        samples, tissues = get_header(data_df_header, num_samples, unknowns)


if __name__ == "__main__":
    main()

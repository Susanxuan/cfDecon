#!/usr/bin/env python

import argparse
import time
import os
import numpy as np
import pandas as pd
from scipy.special import gammaln
import torch
import pickle as pkl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.seterr(divide="ignore", invalid="ignore")


################  support functions   ################


def add_pseudocounts(array, meth):
    """finds values of beta where logll cannot be computed, adds pseudo-counts to make
    computation possible

    array: beta array to check for inproper value
    meth: np array of methylation counts
    """

    idx1 = np.where(
        (array == 0) | (array == 1)
    )
    meth[idx1[0], idx1[1]] += 0.01
    return meth


def check_beta(array):
    """checks for values of beta where log likelihood cannot be computed, returns
    true if can be computed

    array: np array to check
    """

    return (0 in array) or (1 in array)


########  expectation-maximization algorithm  ########


def expectation(beta, alpha):
    """calculates the components needed for log likelihood for each iteration of beta and alpha

    beta: np matrix of the estimated 'true' methylation proportions
    alpha: np matrix of estimated mixing proportions
    """

    e_alpha = alpha.T[:, np.newaxis, :, np.newaxis]
    e_beta = beta[:, :, np.newaxis, :]

    p = e_beta * e_alpha

    p /= np.nansum(p, axis=0)[np.newaxis, ...]

    return p


def log_likelihood(p, x, y, beta, alpha):
    """calculates the log likelihood P(X, Z, Y | alpha, beta)

    p: probability that read has certain read average
    x: input reads
    y: reference reads
    beta: estimated true methylation proportions
    alpha: estimated mixing proportions
    """

    ll_alpha = alpha.T[:, np.newaxis, :]
    ll_beta = beta[:, :, np.newaxis, :]
    ll_y = y[:, :, np.newaxis, :]
    ll_x = np.transpose(x, (1, 0, 2))[np.newaxis, ...]

    ll = 0
    ll += np.sum((ll_y + p * ll_x) * np.log(ll_beta))
    ll += np.sum(np.sum(p * ll_x, axis=3) * np.log(ll_alpha))
    ll += np.sum(gammaln(np.sum(ll_y, axis=3) + 1) - np.sum(gammaln(np.sum(ll_y, axis=3) + 1)))

    return ll


def maximization(p, x, y):
    """maximizes log-likelihood, calculated in the expectation step
    calculates new alpha and beta given these new parameters

    p: probability that read has certain read average
    x: input reads
    y: reference reads
    """

    # in case of overflow or error, transform nans to 0 and inf to large float
    p = np.nan_to_num(p)
    x = np.nan_to_num(x)

    # in p: first index: tissue, second index: sites, third index: individuals
    term1 = p * np.transpose(x, (1, 0, 2))[np.newaxis, ...]
    new_alpha = np.sum(term1, axis=(1, 3)).T
    new_beta = np.sum(term1, axis=2) + y * p.shape[2]

    # check if beta goes out of bounds, if so add pseudocounts to misbehaving y values
    if check_beta(new_beta):
        add_pseudocounts(new_beta, y)
        new_beta = np.sum(term1, axis=2) + y * p.shape[2]

    # return alpha to be normalized to sum to 1
    new_alpha = np.array([row / row.sum() for row in new_alpha])
    new_beta /= np.sum(new_beta, axis=2)[:, :, np.newaxis]
    return new_alpha, new_beta


########################  run em  ########################


def em(x, y, num_iterations, convergence_criteria):

    # randomly intialize alpha for each iteration
    alpha = np.random.uniform(size=(x.shape[0], y.shape[0]))
    alpha /= np.sum(alpha, axis=1)[:, np.newaxis]  # make alpha sum to 1

    # begin by checking for instances where there are no counts for y
    add_pseudocounts(y, y)

    # intialize beta to reference values
    beta = y / np.sum(y, axis=2)[:, :, np.newaxis]

    # perform EM for a given number of iterations
    for i in range(num_iterations):

        p = expectation(beta, alpha)
        a, b = maximization(p, x, y)

        # check convergence of alpha and beta
        alpha_diff = np.mean(abs(a - alpha)) / np.mean(abs(alpha))
        beta_diff = np.mean(abs(b - beta)) / np.mean(abs(beta))

        if alpha_diff + beta_diff < convergence_criteria:  # if convergence criteria, break
            break

        else:  # set current evaluation of alpha and beta
            alpha = a
            beta = b

    ll = log_likelihood(
        p, x, y, beta, alpha
    )

    return alpha, beta, ll


################## read in data #######################


def define_arrays(sample, num_samples, num_unk):
    """
    takes input data matrix- cfDNA and reference, and creates the arrays to run in EM. Adds
    specified number of unknowns to estimate


    sample: pandas dataframe of data (samples and reference). Assumes there is 3 columns (chrom, start, end)
    before the samples and before the reference
    num_samples: number of samples to deconvolve
    num_unk: number of unknowns to estimate
    """
    test = sample.iloc[:, 3: (num_samples) * 5 + 3].values
    train = sample.iloc[:, (num_samples) * 5 + 3 + 3:].values
    num_tissues = train.shape[1] // 5
    # DEL
    print(num_tissues)
    # DEL

    x = np.array(np.split(test, num_samples, axis=1))
    y = np.array(np.split(train, num_tissues, axis=1))

    # add unknowns
    unknown = np.zeros((num_unk, y.shape[1], 5))
    y_unknown = np.append(y, unknown, axis=0)
    true_beta = y_unknown / np.sum(y_unknown, axis=2)[:, :, np.newaxis]

    return (
        np.nan_to_num(x),
        np.nan_to_num(y_unknown),
        np.nan_to_num(true_beta)
    )


def parse_header_names(header, step):
    parsed_header = []

    for i in range(0, len(header), step):
        parsed_header.append(header[i].rsplit("_", 1)[0])

    return parsed_header


def get_header(sample, num_samples, num_unk):
    """
    gets the tissue and sample names to be used in generating an interpretable output file

    sample: dataframe of input data- with header
    num_samples: number of cfDNA samples
    num_unk: number of unknowns to be estimated
    """

    header = list(sample)

    samples = parse_header_names(
        header[3: (num_samples + 3)], 1
    )  # samples are first part of header
    tissues = parse_header_names(
        header[(num_samples) + 3 + 3:], 1
    )  # tissues are second part of header

    unknowns = ["unknown" + str(i) for i in range(1, num_unk + 1)]

    return samples, tissues + unknowns


def write_output(output_file, output_matrix, header, index, unk):
    """
    write estimated methylation proportions and tissue proportions as txt file

    output_file: outputfile name
    output_matrix: celfeer estimate
    header: tissue names
    index: either number of cpgs or number of samples, depending on type of output
    written
    """
    if len(output_matrix.shape) == 3:
        markers, tissues, _ = output_matrix.shape
        output = pd.DataFrame(output_matrix.reshape(markers, tissues * 5))
        output.columns = pd.MultiIndex.from_product([header, [0, 0.25, 0.5, 0.75, 1]])
    else:
        output = pd.DataFrame(output_matrix)
        output.columns = header
    output.insert(
        0, "", index
    )  # insert either the sample names or cpg numbers as first col

    output.to_csv(output_file, sep="\t", index=False)  # save as text file


def generate_sample_array(n, c, sparse=True, sparse_prob=0.5, rare=False, rare_percentage=0.4):
    # Generate random values for each cell in the array
    values = np.random.rand(n, c)

    # Normalize the values to ensure the sum of each row is 1
    row_sums = np.sum(values, axis=1)
    prop = values / row_sums[:, np.newaxis]

    if sparse:
        print("You set sparse as True, some cell's fraction will be zero, the probability is", sparse_prob)
        ## Only partial simulated data is composed of sparse celltype distribution
        for i in range(int(prop.shape[0] * sparse_prob)):
            indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * sparse_prob))
            prop[i, indices] = 0

        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

    if rare:
        print(
            'You will set some cell type fractions are very small (<3%), '
            'these celltype is randomly chosen by percentage you set before.')
        ## choose celltype
        np.random.seed(0)
        indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * rare_percentage))
        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

        for i in range(int(0.5 * prop.shape[0]) + int(int(rare_percentage * 0.5 * prop.shape[0]))):
            prop[i, indices] = np.random.uniform(0, 0.03, len(indices))
            buf = prop[i, indices].copy()
            prop[i, indices] = 0
            prop[i] = (1 - np.sum(buf)) * prop[i] / np.sum(prop[i])
            prop[i, indices] = buf

    return prop


def simulate_arrays(sample, num_samples, num_unk, proportions=None):
    """
    takes input data matrix- cfDNA and reference, and creates the arrays to run in EM. Adds
    specified number of unknowns to estimate


    sample: pandas dataframe of data (samples and reference). Assumes there is 3 columns (chrom, start, end)
    before the samples and before the reference
    num_samples: number of samples to deconvolve
    num_unk: number of unknowns to estimate
    """
    reference_1 = sample.iloc[:, 3: (num_samples) * 5 + 3].values
    reference_2 = sample.iloc[:, (num_samples) * 5 + 3 + 3:].values
    num_tissues = reference_2.shape[1] // 5
    # DEL
    print(num_tissues)
    # DEL
    print(reference_1.shape, reference_2.shape)
    ref_1 = np.array(np.split(reference_1, num_samples, axis=1))
    ref_2 = np.array(np.split(reference_2, num_tissues, axis=1))

    # add unknowns
    unknown = np.zeros((num_unk, ref_2.shape[1], 5))
    y_unknown = np.append(ref_2, unknown, axis=0)

    x_2 = np.array(np.split(reference_2, num_tissues, axis=1))
    tissue_totaldepths = np.sum(x_2, axis=(1, 2))
    x_2 = (x_2.T / tissue_totaldepths).T * np.average(tissue_totaldepths)
    x_2_percents = x_2.reshape(x_2.shape[0], x_2.shape[1] * x_2.shape[2])
    mix_x_2_percents = np.dot(proportions, x_2_percents)
    mix_x_2 = mix_x_2_percents.reshape(-1, x_2.shape[1], 5)
    true_beta_2 = ref_2 / np.sum(ref_2, axis=2)[:, :, np.newaxis]

    return (
        np.nan_to_num(mix_x_2),
        np.nan_to_num(true_beta_2)
    )


def RMSEscore(pred, true):
    return np.mean(np.abs(pred - true))


def CCCscore(y_pred, y_true):
    # pred: shape{n sample, m cell}
    ccc_value = 0
    for i in range(y_pred.shape[1]):
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        # print(r)
        # Mean
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        # Variance
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        # Standard deviation
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        # print(ccc)
        ccc_value += ccc
    return ccc_value / y_pred.shape[1]


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


def plot_scatter_plots(beta_est, beta_truth, input_path, file):
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
    fig.savefig('./results/figure/' + "celfeer" + input_path + "sim2" + '.png')


def simulate_arrays(sample, num_samples, num_unk, proportions=None):
    """
    takes input data matrix- cfDNA and reference, and creates the arrays to run in EM. Adds
    specified number of unknowns to estimate


    sample: pandas dataframe of data (samples and reference). Assumes there is 3 columns (chrom, start, end)
    before the samples and before the reference
    num_samples: number of samples to deconvolve
    num_unk: number of unknowns to estimate
    """
    reference_1 = sample.iloc[:, 3: (num_samples) * 5 + 3].values
    reference_2 = sample.iloc[:, (num_samples) * 5 + 3 + 3:].values
    num_tissues = reference_1.shape[1] // 5
    # DEL
    print(num_tissues)
    # DEL

    ref_1 = np.array(np.split(reference_1, num_samples, axis=1))
    ref_2 = np.array(np.split(reference_2, num_tissues, axis=1))

    # # add unknowns
    # if num_unk:
    #     for col in num_unk:
    #         y[col] = 0
    # unknown = np.zeros((num_unk, ref_2.shape[1], 5))
    # y_unknown = np.append(ref_2, unknown, axis=0)
    x_1 = np.array(np.split(reference_1, num_tissues, axis=1))
    tissue_totaldepths = np.sum(x_1, axis=(1, 2))
    x_1 = (x_1.T / tissue_totaldepths).T * np.average(tissue_totaldepths)
    x_1_percents = x_1.reshape(x_1.shape[0], x_1.shape[1] * x_1.shape[2])
    mix_x_1_percents = np.dot(proportions[:int(proportions.shape[0] / 2), :], x_1_percents)
    mix_x_1 = mix_x_1_percents.reshape(-1, x_1.shape[1], 5)
    true_beta_1 = ref_1 / np.sum(ref_1, axis=2)[:, :, np.newaxis]

    x_2 = np.array(np.split(reference_2, num_tissues, axis=1))
    tissue_totaldepths = np.sum(x_2, axis=(1, 2))
    x_2 = (x_2.T / tissue_totaldepths).T * np.average(tissue_totaldepths)
    x_2_percents = x_2.reshape(x_2.shape[0], x_2.shape[1] * x_2.shape[2])
    mix_x_2_percents = np.dot(proportions[int(proportions.shape[0] / 2):, :], x_2_percents)
    mix_x_2 = mix_x_2_percents.reshape(-1, x_2.shape[1], 5)
    true_beta_2 = ref_2 / np.sum(ref_2, axis=2)[:, :, np.newaxis]
    print('mix_x_1', mix_x_1.shape)
    print('true_beta_2', true_beta_2.shape)
    mix_x = np.concatenate([mix_x_1, mix_x_2])
    true_beta = np.mean([true_beta_1, true_beta_2], axis=0)
    y = np.mean([ref_1, ref_2], axis=0)
    # print('mix_x',mix_x.shape)
    # print('y',y.shape)
    # print('true_beta',mix_x.shape)
    return (
        np.nan_to_num(mix_x),
        np.nan_to_num(y),
        np.nan_to_num(true_beta)
    )


################## run #######################

if __name__ == "__main__":
    startTime = time.time()
    #input_path = "WGBS_sim_input.txt"
    output_directory = "./results"
    num_samples = 7  # 52
    unknowns = 0
    parallel_job_id = 1
    mode = "overall"  # high-resolution(dictionary  overall
    np.random.seed(parallel_job_id)
    unknowns_list = [int(x) for x in unknowns.split(",")] if unknowns else None
    num_tissues = 7
    train = 1
    simi = 2
    test = 0
    max_iterations = 1000
    convergence = 0.0001
    random = 10

    # make output directory if it does not exist
    if not os.path.exists(output_directory) and parallel_job_id == 1:
        os.makedirs(output_directory)
        print("made " + output_directory + "/")
        print()
    else:
        print("writing to " + output_directory + "/")

    # make input arrays and add the specified number of unknowns
    # x, y = define_arrays(data_df, int(num_samples), int(unknowns))  # 这里就是ref1 和 ref2
    # prop只和train有关
    for input_path in ["WGBS_sim_input.txt"]:
        data_df = pd.read_csv(
            input_path, delimiter="\t", header=None, skiprows=1)  # read input samples/reference data
        for file in ["0.1False0.4", "0.3False0.4", "0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"]:  # "0.3True0.2"
            if train == 1:
                print(input_path, file)
                sparse, ifrare, rare = extract_values(file)
                proportions = "./results/fractions/sample_array_10007_spar" + file + ".pkl"  #
                props = pkl.load(open(proportions, 'rb'))

                x, y, true_beta = simulate_arrays(data_df, num_samples, unknowns, proportions=props)

                data = torch.tensor(x)
                label = torch.tensor(props)

                train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.1, random_state=0)
                test_x_numpy = test_x.numpy()
                # Run EM with the specified iterations and convergence criteria
                random_restarts = []

                for i in range(random):
                    alpha, beta, ll = em(
                        test_x_numpy, y, max_iterations, convergence
                    )
                    random_restarts.append((ll, alpha, beta))

                ll_max, alpha_max, beta_max = max(random_restarts)  # pick best random restart per replicate
                # elapsed_time = time() - startTime
                # print('Total time: %d seconds.' % int(elapsed_time))
                l1, ccc = score(alpha_max, test_y.cpu().detach().numpy())
                print('l1, ccc:', l1, ccc)

                with open(output_directory + "/" + "celfeer" + input_path + str(simi) + mode + file + "alpha_est.pkl", "wb") as f:
                    pkl.dump(alpha_max, f)
                with open(output_directory + "/" + "celfeer" + input_path + str(simi) + mode + file + "alpha_true.pkl", "wb") as f:
                    pkl.dump(props, f)
                with open(output_directory + "/" + "celfeer" + input_path + str(simi) + mode + file + "beta_est.pkl", "wb") as f:
                    pkl.dump(beta_max, f)
                with open(output_directory + "/" + "celfeer" + input_path + str(simi) + mode + file + "beta_true.pkl", "wb") as f:
                    pkl.dump(true_beta, f)

        if test == 1:
            x, y, true_beta = define_arrays(data_df, int(num_samples), int(unknowns))

            random_restarts = []

            for i in range(random):
                alpha, beta, ll = em(
                    x, y, max_iterations, convergence
                )
                random_restarts.append((ll, alpha, beta))

            ll_max, alpha_max, beta_max = max(
                random_restarts
            )

            with open(output_directory + "/test" + "celfeer" + input_path + str(simi) + mode + "alpha_est.pkl", "wb") as f:
                pkl.dump(alpha_max, f)
            with open(output_directory + "/test" + "celfeer" + input_path + str(simi) + mode + "beta_est.pkl", "wb") as f:
                pkl.dump(beta_max, f)
            with open(output_directory + "/test" + "celfeer" + input_path + str(simi) + mode + "beta_true.pkl", "wb") as f:
                pkl.dump(true_beta, f)

            beta_truth = true_beta
            beta_est = beta_max
            beta_est = process_beta(beta_truth, beta_est)

            calculate_scores(beta_est, beta_truth)
            plot_scatter_plots(beta_est, beta_truth, input_path, file)

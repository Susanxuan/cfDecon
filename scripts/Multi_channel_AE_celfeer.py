import argparse

import numpy as np
import pandas as pd
import os
from scipy.special import gammaln
import pickle as pkl
from tqdm import tqdm
import anndata
from anndata import read_h5ad
import warnings
import time

warnings.filterwarnings("ignore")
import torch
import math
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from numpy.random import choice
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from train_model import reproducibility, test_AE_function, predict_AE_function
from utils import generate_sample_array, new_generate_sample_array, extract_values


# ################  support functions   ################


# def add_pseudocounts(array, meth):
#     """finds values of beta where logll cannot be computed, adds pseudo-counts to make
#     computation possible

#     array: beta array to check for inproper value
#     meth: np array of methylation counts
#     """

#     idx1 = np.where(
#         (array == 0) | (array == 1)
#     )
#     meth[idx1[0], idx1[1]] += 0.01
#     return meth


# def check_beta(array):
#     """checks for values of beta where log likelihood cannot be computed, returns
#     true if can be computed

#     array: np array to check
#     """

#     return (0 in array) or (1 in array)

# ########  expectation-maximization algorithm  ########


# def expectation(beta, alpha):
#     """calculates the components needed for log likelihood for each iteration of beta and alpha

#     beta: np matrix of the estimated 'true' methylation proportions
#     alpha: np matrix of estimated mixing proportions
#     """

#     e_alpha = alpha.T[:, np.newaxis, :, np.newaxis]
#     e_beta = beta[:, :, np.newaxis, :]

#     p = e_beta * e_alpha

#     p /= np.nansum(p, axis=0)[np.newaxis, ...]

#     return p


# def log_likelihood(p, x, y, beta, alpha):
#     """calculates the log likelihood P(X, Z, Y | alpha, beta)

#     p: probability that read has certain read average
#     x: input reads
#     y: reference reads
#     beta: estimated true methylation proportions
#     alpha: estimated mixing proportions
#     """

#     ll_alpha = alpha.T[:, np.newaxis, :]
#     ll_beta = beta[:, :, np.newaxis, :]
#     ll_y = y[:, :, np.newaxis, :]
#     ll_x = np.transpose(x, (1, 0, 2))[np.newaxis, ...]

#     ll = 0
#     ll += np.sum((ll_y + p * ll_x) * np.log(ll_beta))
#     ll += np.sum(np.sum(p * ll_x, axis=3) * np.log(ll_alpha))
#     ll += np.sum(gammaln(np.sum(ll_y, axis=3) + 1) - np.sum(gammaln(np.sum(ll_y, axis=3) + 1)))

#     return ll


# def maximization(p, x, y):
#     """maximizes log-likelihood, calculated in the expectation step
#     calculates new alpha and beta given these new parameters

#     p: probability that read has certain read average
#     x: input reads
#     y: reference reads
#     """

#     # in case of overflow or error, transform nans to 0 and inf to large float
#     p = np.nan_to_num(p)
#     x = np.nan_to_num(x)

#     # in p: first index: tissue, second index: sites, third index: individuals
#     term1 = p * np.transpose(x, (1, 0, 2))[np.newaxis, ...]
#     new_alpha = np.sum(term1, axis=(1, 3)).T
#     new_beta = np.sum(term1, axis=2) + y * p.shape[2]

#     # check if beta goes out of bounds, if so add pseudocounts to misbehaving y values
#     if check_beta(new_beta):
#         add_pseudocounts(new_beta, y)
#         new_beta = np.sum(term1, axis=2) + y * p.shape[2]

#     # return alpha to be normalized to sum to 1
#     new_alpha = np.array([row / row.sum() for row in new_alpha])
#     new_beta /= np.sum(new_beta, axis=2)[:, :, np.newaxis]
#     return new_alpha, new_beta


# ########################  run em  ########################


# def em(x, y, num_iterations, convergence_criteria):
#     """take in the input cfdna matrices and the reference data and
#     runs the EM for the specified number of iterations, or stops once the
#     convergence_criteria is reached

#     x: input reads
#     y: reference reads
#     convergence_criteria: difference between alpha + beta before stopping
#     """

#     # randomly intialize alpha for each iteration
#     alpha = np.random.uniform(size=(x.shape[0], y.shape[0]))
#     alpha /= np.sum(alpha, axis=1)[:, np.newaxis]  # make alpha sum to 1

#     # begin by checking for instances where there are no counts for y
#     add_pseudocounts(y, y)

#     # intialize beta to reference values
#     beta = y / np.sum(y, axis=2)[:, :, np.newaxis]

#     # perform EM for a given number of iterations
#     for i in range(num_iterations):

#         p = expectation(beta, alpha)
#         a, b = maximization(p, x, y)

#         # check convergence of alpha and beta
#         alpha_diff = np.mean(abs(a - alpha)) / np.mean(abs(alpha))
#         beta_diff = np.mean(abs(b - beta)) / np.mean(abs(beta))

#         if alpha_diff + beta_diff < convergence_criteria:  # if convergence criteria, break
#             break

#         else:  # set current evaluation of alpha and beta
#             alpha = a
#             beta = b

#     ll = log_likelihood(
#         p, x, y, beta, alpha
#     )

#     return alpha, beta, ll

################## read in data #######################


def define_arrays(sample, num_samples, num_unk, proportions=None):
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

    test_x = np.array(np.split(test, num_samples, axis=1))
    y = np.array(np.split(train, num_tissues, axis=1))

    # add unknowns
    unknown = np.zeros((num_unk, y.shape[1], 5))
    y_unknown = np.append(y, unknown, axis=0)
    if proportions is not None:
        x = np.array(np.split(train, num_tissues, axis=1))
        tissue_totaldepths = np.sum(x, axis=(1, 2))
        x = (x.T / tissue_totaldepths).T * np.average(tissue_totaldepths)

        x_percents = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

        mix_x_percents = np.dot(proportions, x_percents)

        mix_x = mix_x_percents.reshape(proportions.shape[0], x.shape[1], 5)

        true_beta = y / np.sum(y, axis=2)[:, :, np.newaxis]
        return (
            np.nan_to_num(test_x),
            np.nan_to_num(y_unknown),
            np.nan_to_num(mix_x),
            np.nan_to_num(true_beta)
        )
    else:

        return (
            np.nan_to_num(test_x),
            np.nan_to_num(y_unknown),
        )


def simulate_arrays(sample, num_samples, num_unk, proportions=None, simi=1, sparse=None, ifrare=None, rare=None):
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
    if proportions is not None:
        if simi == 2:
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

            return (
                np.nan_to_num(mix_x_1),
                np.nan_to_num(true_beta_1),
                np.nan_to_num(mix_x_2),
                np.nan_to_num(true_beta_2)
            )
        else:
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

    else:
        training_props = generate_sample_array(5000, num_tissues, sparse=True, sparse_prob=sparse,
                                               rare=(ifrare == str(True)), rare_percentage=rare)
        x_2 = np.array(np.split(reference_2, num_tissues, axis=1))
        tissue_totaldepths = np.sum(x_2, axis=(1, 2))
        x_2 = (x_2.T / tissue_totaldepths).T * np.average(tissue_totaldepths)
        x_2_percents = x_2.reshape(x_2.shape[0], x_2.shape[1] * x_2.shape[2])
        mix_x_2_percents = np.dot(training_props, x_2_percents)
        mix_x_2 = mix_x_2_percents.reshape(-1, x_2.shape[1], 5)
        true_beta_2 = ref_2 / np.sum(ref_2, axis=2)[:, :, np.newaxis]
        return (
            np.nan_to_num(mix_x_2),
            np.nan_to_num(true_beta_2),
            np.nan_to_num(ref_1),
            training_props
        )



def simulate_arrays_new(sample, num_samples, num_unk, proportions=None, simi=1, n=None, fix_c=None):
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
    #print(num_tissues)
    # DEL
    #print(reference_1.shape, reference_2.shape)
    ref_1 = np.array(np.split(reference_1, num_samples, axis=1))
    ref_2 = np.array(np.split(reference_2, num_tissues, axis=1))

    # add unknowns
    unknown = np.zeros((num_unk, ref_2.shape[1], 5))
    y_unknown = np.append(ref_2, unknown, axis=0)
    if proportions is not None:
        if simi == 2:
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

            return (
                np.nan_to_num(mix_x_1),
                np.nan_to_num(true_beta_1),
                np.nan_to_num(mix_x_2),
                np.nan_to_num(true_beta_2)
            )
        else:
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

    else:
        training_props = new_generate_sample_array(n, c=31, sparse=False, sparse_prob=0.3, rare=False, rare_percentage=0.4,
                                                   WBC_major=False,
                                                   UXM=True, fix_c=(fix_c == str(True)))
        x_2 = np.array(np.split(reference_2, num_tissues, axis=1))
        tissue_totaldepths = np.sum(x_2, axis=(1, 2))
        x_2 = (x_2.T / tissue_totaldepths).T * np.average(tissue_totaldepths)
        x_2_percents = x_2.reshape(x_2.shape[0], x_2.shape[1] * x_2.shape[2])
        mix_x_2_percents = np.dot(training_props, x_2_percents)
        mix_x_2 = mix_x_2_percents.reshape(-1, x_2.shape[1], 5)
        true_beta_2 = ref_2 / np.sum(ref_2, axis=2)[:, :, np.newaxis]
        return (
            np.nan_to_num(mix_x_2),
            np.nan_to_num(true_beta_2),
            np.nan_to_num(ref_1),
            training_props
        )


def simulate_arrays_miss(sample, num_samples, num_unk, proportions=None, simi=1, sparse=None, ifrare=None, rare=None,
                         miss=None, training_props=None):
    """
    takes input data matrix- cfDNA and reference, and creates the arrays to run in EM. Adds
    specified number of unknowns to estimate


    sample: pandas dataframe of data (samples and reference). Assumes there is 3 columns (chrom, start, end)
    before the samples and before the reference
    num_samples: number of samples to deconvolve
    num_unk: number of unknowns to estimate
    """
    reference_1 = sample.iloc[:, 3: (num_samples) * 5 + 3].values
    ref_1 = np.array(np.split(reference_1, num_samples, axis=1))

    reference_2_ori = sample.iloc[:, (num_samples) * 5 + 3 + 3:].values
    num_tissues = reference_2_ori.shape[1] // 5  # 19
    ref_2_ori = np.array(np.split(reference_2_ori, num_tissues, axis=1))  # n*(19*5)
    # miss是个数
    indices = np.array(range(reference_2_ori.shape[1]))
    delete_indices = np.arange(miss * 5)
    keep_indices = np.delete(indices, delete_indices)
    reference_2 = reference_2_ori[:, keep_indices]

    # miss是索引
    # miss_idx = miss
    # indices = np.array(range(reference_2.shape[1]))
    # keep_indices = np.delete(indices, np.arange(miss_idx * 5, (miss_idx + 1) * 5))
    # reference_2 = reference_2[:, keep_indices]

    ref_2 = np.array(np.split(reference_2, num_tissues - miss, axis=1))

    # add unknowns
    unknown = np.zeros((num_unk, ref_2.shape[1], 5))
    y_unknown = np.append(ref_2, unknown, axis=0)
    if proportions is not None:
        if simi == 2:
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

            return (
                np.nan_to_num(mix_x_1),
                np.nan_to_num(true_beta_1),
                np.nan_to_num(mix_x_2),
                np.nan_to_num(true_beta_2)
            )
        else:
            x_2 = np.array(np.split(reference_2_ori, num_tissues, axis=1))
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

    else:
        x_2 = np.array(np.split(reference_2, num_tissues, axis=1))
        tissue_totaldepths = np.sum(x_2, axis=(1, 2))
        x_2 = (x_2.T / tissue_totaldepths).T * np.average(tissue_totaldepths)
        x_2_percents = x_2.reshape(x_2.shape[0], x_2.shape[1] * x_2.shape[2])
        mix_x_2_percents = np.dot(training_props, x_2_percents)
        mix_x_2 = mix_x_2_percents.reshape(-1, x_2.shape[1], 5)
        true_beta_2 = ref_2 / np.sum(ref_2, axis=2)[:, :, np.newaxis]
        return (
            np.nan_to_num(mix_x_2),
            np.nan_to_num(true_beta_2),
            np.nan_to_num(ref_1),
            training_props
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


################## run #######################

if __name__ == "__main__":
    startTime = time.time()
    # read command line input parameters
    parser = argparse.ArgumentParser(
        description="CelFEER - Read resolution adaptation of CelFiE: Cell-free DNA decomposition."
    )
    parser.add_argument("input_path", help="the path to the input file")
    parser.add_argument("output_directory", help="the path to the output directory")
    parser.add_argument("num_samples", type=int, help="Number of cfdna samples")
    parser.add_argument("num_celltypes", type=int, help="Number of reference cell types")
    parser.add_argument(
        "-u",
        "--unknowns",
        default=0,
        type=int,
        help="Number of unknown categories to be estimated along with the reference data. Default 0.",
    )
    parser.add_argument(
        "-p",
        "--parallel_job_id",
        default=1,
        type=int,
        help="Replicate number in a simulation experiment. Default 1. ",
    )
    parser.add_argument(
        "-c",
        "--convergence",
        default=0.0001,
        type=float,
        help="Convergence criteria for EM. Default 0.0001.",
    )
    parser.add_argument(
        "-r",
        "--random_restarts",
        default=10,
        type=int,
        help="CelFEER will perform several random restarts and select the one with the highest log-likelihood. Default 10.",
    )
    parser.add_argument(
        "-f",
        "--proportions",
        default=None,
        type=str,
        help="Pickle file containing the tissue mixture proportions. Put None if the proportions are hard coded.",
    )
    parser.add_argument(
        "-s",
        "--simi",
        default='1',
        type=str,
        help="simi mode",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default='overall',
        type=str,
        help="adaptive stage: high-resolution or overall",
    )
    parser.add_argument(
        "-a", "--adaptive", dest="adaptive", action="store_true",
        help="Whether use adaptive stage"
    )
    args = parser.parse_args()

    # make output directory if it does not exist
    if not os.path.exists(args.output_directory) and args.parallel_job_id == 1:
        os.makedirs(args.output_directory)
        print("made " + args.output_directory + "/")
        print()
    else:
        print("writing to " + args.output_directory + "/")

    data_df = pd.read_csv(
        args.input_path, delimiter="\t", header=None, skiprows=1)  # read input samples/reference data

    print(f"finished reading {args.input_path}")
    print()

    output_alpha_file = f"{args.output_directory}/{args.parallel_job_id}_tissue_proportions.txt"
    output_beta_file = f"{args.output_directory}/{args.parallel_job_id}_methylation_proportions.txt"

    print(f"beginning generation of {args.output_directory}")
    print()

    # Load file containing proportions 
    if args.proportions:
        props = pkl.load(open(args.proportions, 'rb'))
        # make input arrays and add the specified number of unknowns
        # x, reference, simulated_x, true_beta = define_arrays(data_df, int(args.num_samples), int(args.unknowns),proportions=props)
        # Split the data into training and testing sets
        # train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.1, random_state=0)
        if args.simi == '2':
            # simulated 2 data
            simulated_x_1, true_beta_1, simulated_x_2, true_beta_2 = simulate_arrays(data_df, int(args.num_samples),
                                                                                     int(args.unknowns),
                                                                                     proportions=props, simi=args.simi)
            data_1 = torch.tensor(simulated_x_1)
            label_1 = torch.tensor(props[:int(props.shape[0] / 2), :])
            data_2 = torch.tensor(simulated_x_2)
            label_2 = torch.tensor(props[int(props.shape[0] / 2):, :])
            train_x_1, test_x_1, train_y_1, test_y_1 = train_test_split(data_1, label_1, test_size=0.1, random_state=0)
            train_x_2, test_x_2, train_y_2, test_y_2 = train_test_split(data_2, label_2, test_size=0.1, random_state=0)

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
            x, true_beta = simulate_arrays(data_df, int(args.num_samples), int(args.unknowns), proportions=props,
                                           simi=args.simi)
            data = torch.tensor(x)
            label = torch.tensor(props)
            train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.1, random_state=0)
        L1 = []
        CCC = []
        if args.adaptive:
            l1, ccc, pred, sign = test_AE_function(train_x, train_y, test_x, test_y, outdir=args.output_directory,
                                                   adaptive=args.adaptive, mode=args.mode)
            print('l1, ccc:', l1, ccc)
            L1.append(l1)
            CCC.append(ccc)
            with open(args.output_directory + "/alpha_est.pkl", "wb") as f:
                pkl.dump(pred, f)
            with open(args.output_directory + "/alpha_true.pkl", "wb") as f:
                pkl.dump(props, f)
            with open(args.output_directory + "/beta_est.pkl", "wb") as f:
                pkl.dump(sign, f)

            if args.simi == '2':
                with open(args.output_directory + "/beta_true_1.pkl", "wb") as f:
                    pkl.dump(true_beta_1, f)
                with open(args.output_directory + "/beta_true_2.pkl", "wb") as f:
                    pkl.dump(true_beta_2, f)
            else:
                with open(args.output_directory + "/beta_true.pkl", "wb") as f:
                    pkl.dump(true_beta, f)
        else:
            l1, ccc, pred = test_AE_function(train_x, train_y, test_x, test_y, outdir=args.output_directory,
                                             adaptive=args.adaptive, mode=args.mode)
            print('l1, ccc:', l1, ccc)
            L1.append(l1)
            CCC.append(ccc)
            with open(args.output_directory + "/alpha_est.pkl", "wb") as f:
                pkl.dump(pred, f)
            with open(args.output_directory + "/alpha_true.pkl", "wb") as f:
                pkl.dump(props, f)
    else:
        x, true_beta, test_samples, training_props = simulate_arrays(data_df, int(args.num_samples), int(args.unknowns))
        data = torch.tensor(x)
        label = torch.tensor(training_props)

        # #kfold to test the model performance before predict the real data
        # kfold = KFold(n_splits=5,shuffle=True,random_state=0).split(data,label)

        # for k, (train_indices, test_indices) in enumerate(kfold):
        #     reproducibility(seed=0)
        #     # print(train_indices,test_indices)
        #     train_x = data[train_indices]
        #     train_y = label[train_indices]

        #     test_x = data[test_indices]
        #     test_y = label[test_indices]

        #     l1, ccc, pred = test_AE_function(train_x,train_y,test_x,test_y,outdir=args.output_directory,adaptive=args.adaptive, mode=args.mode)
        #     print('l1, ccc:',l1, ccc)

        if args.adaptive:
            pred, sigm = predict_AE_function(data, label, test_samples, int(args.num_celltypes),
                                             outdir=args.output_directory, adaptive=args.adaptive, mode=args.mode)
            with open(args.output_directory + "/alpha_est.pkl", "wb") as f:
                pkl.dump(pred, f)
            print(pred)
            # pred.to_csv(args.output_directory + "/alpha_est.csv")
            with open(args.output_directory + "/beta_est.pkl", "wb") as f:
                pkl.dump(sigm, f)
            with open(args.output_directory + "/beta_true.pkl", "wb") as f:
                pkl.dump(true_beta, f)
            data_df_header = pd.read_csv(
                args.input_path, delimiter="\t", nrows=1
            )
            # get header for output files
            samples, tissues = get_header(data_df_header, args.num_samples, args.unknowns)

            # write estimates as text files
            # output_alpha_file=args.output_directory + "/alpha_est.csv"
            # write_output(output_alpha_file, pred, tissues, samples, args.unknowns)
        else:
            pred = predict_AE_function(data, label, test_samples, int(args.num_celltypes), outdir=args.output_directory,
                                       adaptive=args.adaptive, mode=args.mode)
            # print(pred)
            # pred.to_csv(args.output_directory + "/alpha_est.csv")
            with open(args.output_directory + "/alpha_est.pkl", "wb") as f:
                pkl.dump(pred, f)

            data_df_header = pd.read_csv(
                args.input_path, delimiter="\t", nrows=1
            )
            # get header for output files
            samples, tissues = get_header(data_df_header, args.num_samples, args.unknowns)

            # write estimates as text files
            output_alpha_file = args.output_directory + "/alpha_est.csv"
            write_output(output_alpha_file, pred, tissues, samples, args.unknowns)

#!/usr/bin/env python

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


np.seterr(divide="ignore", invalid="ignore")


## Functions


def generate_beta(i, j):
    """
    randomly generates a matrix of methylation values between 0 and 1
    tissue: number of tissues
    cpg: number of cpgs
    """
    beta = np.random.uniform(size=(i, j, 5))
    beta /= np.sum(beta, axis=2)[:, :, np.newaxis]
    return beta


def generate_reads(alpha, beta, x_depths):
    """
    generates the cfDNA reads based on the generated tissue proportions, the true
    methylation values in the reference and the depths

    alpha: tissue props
    beta: methylation values for the reference
    x_depths: simulated read depths for each CpG in each individual in cfDNA input
    """

    total_indiv = alpha.shape[0]
    i, total_cpg = beta.shape[0], beta.shape[1]

    reads = np.zeros((total_indiv, total_cpg, 5))

    for n in range(total_indiv):
        for j in range(total_cpg):
            depth = x_depths[n, j]  # depth at a paticular cpg and person
            beta_cpg = beta[:, j]  # "true" methylation value in the reference

            mix = np.random.choice(
                i,
                depth,
                replace=True,
                p=alpha[
                    n,
                ],
            )  # assign reads based on the tissue proportions for that individual
            probability = beta_cpg[mix]

            reads[n, j] = np.sum(
                np.random.binomial(1, probability, size=(depth, 5)), axis=0
            )  # the beta is the sum of all the individual reads coming from the tissues contributing to that cpg in that individual

    return reads


def generate_counts(count, probability):
    """
    generate the methylation read counts for the reference data

    count: read depths
    probability: probability of each cpg being methylated for each tissue
    """
    counts = np.zeros(probability.shape)
    for i in range(count.shape[0]):
        for j in range(count.shape[1]):
            counts[i, j, :] = np.random.multinomial(count[i, j], probability[i, j, :])
    return counts


def generate_depths(depth, input_shape):
    """
    creates array of ints where each number represents the number of total reads in a tissue at a cpg

    depth: read depth
    input_shape: number of tissues X number of CpGs
    """

    return np.random.poisson(depth, input_shape)


def rare_cell_type(tissues, individuals):
    """
    creates true cell type proportions for a group of individuals where half has cfDNA containing a rare cell type
    and the other half does not

    tissues: number of tissues
    individuals: number of individuals
    """
    alpha_int_1 = np.zeros(tissues)
    alpha_int_1[0] = 0.01
    alpha_int_1[1:] = np.random.uniform(0.5, 1, size=tissues - 1)
    alpha_int_1[1:] = alpha_int_1[1:] * 0.99 / alpha_int_1[1:].sum()
    alpha_int_1 = alpha_int_1.reshape(1, tissues)
    alpha_int_1 = np.repeat(alpha_int_1, individuals / 2, axis=0)  # repeat for the number of individuals

    alpha_int_2 = np.zeros(tissues)
    alpha_int_2[1:] = np.random.uniform(0.5, 1, size=tissues - 1)
    alpha_int_2[1:] = alpha_int_2[1:] / alpha_int_2[1:].sum()
    alpha_int_2 = alpha_int_2.reshape(1, tissues)
    alpha_int_2 = np.repeat(alpha_int_2, individuals / 2, axis=0)  # repeat for the number of individuals
    alpha_int = np.vstack((alpha_int_1, alpha_int_2))
    return alpha_int


def missing_cell_type(tissues, individuals):
    """
    creates true cell type proportions drawn from a normal distribution, where one cell type in the reference data
    does not appear in the cfdna of any individual

    tissues: number of tissues
    individuals: number of individuals
    """
    alpha_int = np.zeros(tissues)
    alpha_int[-1] = np.clip(np.random.normal(0.2, 0.1), 0, 1)
    alpha_int[:-1] = np.random.uniform(0, 1, size=tissues - 1)
    alpha_int = alpha_int / alpha_int.sum()
    alpha_int = alpha_int.reshape(1, tissues)
    alpha_int = np.repeat(alpha_int, individuals, axis=0)  # repeat for the number of individuals
    return alpha_int


def generate_em_replicate(tissues, cpgs, individuals, depth, beta_depth, unknowns):
    """
    generates the input data for the simulation experiments according to the distributions described in paper

    tissues: number of tissues
    cpgs: number of cpgs
    individuals: number of individuals
    depth: expected depth of cfdna data
    beta_depth: expected depth of reference data
    unknown: number of unknown cell types
    """
    # let alpha be a simple increasing vector of proportions
    alpha_int = np.array(list(range(1, tissues + 1)))
    alpha_int = (alpha_int / alpha_int.sum()).reshape(1, len(alpha_int))
    alpha_int = np.repeat(alpha_int, individuals, axis=0)  # repeat for the number of individuals

    # comment out for replicating experiments with missing cell types
    # alpha_int = missing_cell_type(tissues, individuals)

    # comment out for replicating experiments with rare cell type
    # alpha_int = rare_cell_type(tissues, individuals)

    beta_int = generate_beta(tissues, cpgs)

    Y_int_depths = generate_depths(beta_depth, (tissues, cpgs))
    Y_int = generate_counts(Y_int_depths, beta_int)

    X_int_depths = generate_depths(depth, (individuals, cpgs))
    X_int = generate_reads(alpha_int, beta_int, X_int_depths)

    # set unknowns
    if unknowns > 0:
        Y_int[-unknowns:] = 0

    return Y_int, X_int, alpha_int, beta_int

def add_pseudocounts(array, meth):
    """
    finds values of beta where logll cannot be computed, adds pseudo-counts to make
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
    """
    checks for values of beta where log likelihood cannot be computed, returns
    true if can be computed

    array: np array to check
    whether there is a 0 or 1 in the array
    """

    return (0 in array) or (1 in array)


########  expectation-maximization algorithm  ########


def expectation(beta, alpha):
    """
    calculates the components needed for log likelihood for each iteration of beta and alpha

    beta: np matrix of the estimated 'true' methylation proportions
    alpha: np matrix of estimated mixing proportions
    """

    e_alpha = alpha.T[:, np.newaxis, :, np.newaxis]
    e_beta = beta[:, :, np.newaxis, :]

    p = e_beta * e_alpha

    p /= np.nansum(p, axis=0)[np.newaxis, ...]

    return p


def log_likelihood(p, x, y, beta, alpha):
    """
    calculates the log likelihood P(X, Z, Y | alpha, beta)

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
    """
    maximizes log-likelihood, calculated in the expectation step
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


def em(x, y, num_iterations, convergence_criteria):
    """
    take in the input cfdna matrices and the reference data and
    runs the EM for the specified number of iterations, or stops once the
    convergence_criteria is reached

    x: input reads
    y: reference reads
    convergence_criteria: difference between alpha + beta before stopping
    """

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

################## AE ########################

class simdatset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # x = torch.from_numpy(self.X[index]).float().to(device)
        # y = torch.from_numpy(self.Y[index]).float().to(device)
        x = torch.as_tensor(self.X[index], dtype=torch.float32, device=device)
        y = torch.as_tensor(self.Y[index], dtype=torch.float32, device=device)
        return x, y
    
class ChannelEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ChannelEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_dim, 512),
            nn.CELU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.CELU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.CELU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.CELU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class MultiChannelEncoder(nn.Module):
    def __init__(self, input_dim, channels, output_dim):
        super(MultiChannelEncoder, self).__init__()
        self.channels = channels
        self.encoders = nn.ModuleList([ChannelEncoder(input_dim, output_dim) for _ in range(channels)])
        self.compressor = nn.Sequential(
            nn.Linear(output_dim * channels, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        encoded_outputs = []
        for i in range(self.channels):
            # print('x[:,i]',x[:,:,i].shape,x.shape)
            # print(self.encoders[i])
            encoded_output = self.encoders[i](x[:,:,i])
            encoded_outputs.append(encoded_output)
        stacked_tensor = torch.stack(encoded_outputs, dim=1)
        compressed_tensor = self.compressor(stacked_tensor.view(stacked_tensor.size(0), -1))
        # print('stacked_tensor',stacked_tensor.shape,'compressed_tensor',compressed_tensor.shape)
        return compressed_tensor
    

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, channels, output_dim):
        super().__init__()
        self.name = 'ae'
        self.inputdim = input_dim
        self.channels = channels
        self.outputdim = output_dim
        # self.encoder = nn.Sequential(nn.Dropout(),
        #                              nn.Linear(self.inputdim*self.channels,512),
        #                              nn.CELU(),
        #                              nn.Dropout(),
        #                              nn.Linear(512,256),
        #                              nn.CELU(),
        #                              nn.Dropout(),
        #                              nn.Linear(256,128),
        #                              nn.CELU(),
        #                              nn.Dropout(),
        #                              nn.Linear(128,64),
        #                              nn.CELU(),
        #                              nn.Linear(64,output_dim),
        #                              nn.Softmax()
        #                              )
        self.encoder = MultiChannelEncoder(self.inputdim, self.channels, self.outputdim)
        
        self.decoder = nn.Sequential(nn.Linear(self.outputdim,64,bias=False),
                                     nn.Linear(64,128,bias=False),
                                     nn.Linear(128,256,bias=False),
                                     nn.Linear(256,512,bias=False),
                                     nn.Linear(512,self.inputdim*self.channels,bias=False))
        
        
                                               
    def encode(self,x):
        # encode_temp=x.view(x.size(0), -1)
        encode_temp=x
        return self.encoder(encode_temp)
    
    def decode(self,z):
        return self.decoder(z).view(z.size(0), -1, self.channels)

    def sigmatrix(self):
        w0 = self.decoder[0].weight.T
        w1 = self.decoder[1].weight.T
        w2 = self.decoder[2].weight.T
        w3 = self.decoder[3].weight.T
        w4 = self.decoder[4].weight.T
        w01 = torch.mm(w0,w1)
        w02 = torch.mm(w01,w2)
        w03 = torch.mm(w02,w3)
        w04 = torch.mm(w03,w4)
        return F.hardtanh(w04,0,1)

    def forward(self, x):
        sigmatrix = self.sigmatrix()
        z = self.encode(x)
        # print('z',z.shape)
        # print('sigmatrix',sigmatrix.shape)
        x_recon = torch.mm(z,sigmatrix)
        return x_recon.view(x_recon.size(0), -1, self.channels), z, sigmatrix



def train(model, train_loader, optimizer, epochs=1):
    model.train()
    loss = []
    recon_loss = []
    for i in tqdm(range(epochs)):
        for k, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            x_recon, cell_prop, sigm = model(data)
            batch_loss = F.l1_loss(cell_prop, label)+F.l1_loss(x_recon,data)
            batch_loss.backward()
            optimizer.step()
            loss.append(F.l1_loss(cell_prop, label).cpu().detach().numpy())
            recon_loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

    return model, loss, recon_loss



def predict(model, data):
    model.eval()
    # data = torch.from_numpy(data).float().to(device)
    data = torch.as_tensor(data, dtype=torch.float32, device=device)
    _, pred, sigmatrix = model(data)
    pred = pred.cpu().detach().numpy()
    sigmatrix = sigmatrix.cpu().detach().numpy()
    return pred, sigmatrix


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

def score(pred, label):
    distance = []
    ccc = []
    new_pred = pred.reshape(-1,1)
    new_label = label.reshape(-1,1)
    distance.append(RMSEscore(new_pred, new_label))
    ccc.append(CCCscore(new_pred, new_label))
    return distance[0], ccc[0]

def showloss(loss):
    plt.figure()
    plt.plot(loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('/mnt/nas/user/yixuan/cfDNA/CelFEER/output/AE_sim/loss.png')
            
def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rates):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._mlp()

    def forward(self,x):
        # x: (n sample, m gene)
        # output: (n sample, k cell proportions)
        return self.model(x)

    def _mlp(self):
        mlp = nn.Sequential(nn.Linear(self.input_dim,self.hidden_units[0]),
                            nn.Dropout(self.dropout_rates[0]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
                            nn.Dropout(self.dropout_rates[1]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_units[1], self.hidden_units[2]),
                            nn.Dropout(self.dropout_rates[2]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_units[2], self.hidden_units[3]),
                            nn.Dropout(self.dropout_rates[3]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_units[3], self.output_dim),
                            nn.Softmax(dim=1)
                            )
        return mlp
def initialize_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data,0)
class scaden():
    def __init__(self, architectures, train_x, train_y, lr=1e-4, batch_size=128, epochs=20):
        self.architectures = architectures
        self.model512 = None
        self.model256 = None
        self.model1024 = None
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.inputdim = train_x.shape[1]
        self.outputdim = train_y.shape[1]
        self.train_loader = DataLoader(simdatset(train_x, train_y), batch_size=batch_size, shuffle=True)


    def _subtrain(self, model, optimizer):
        model.train()
        i = 0
        loss = []
        for i in tqdm(range(self.epochs)):
            for data, label in self.train_loader:
                optimizer.zero_grad()
                batch_loss = F.l1_loss(model(data), label)
                batch_loss.backward()
                optimizer.step()
                loss.append(batch_loss.cpu().detach().numpy())
        return model, loss
    
    def showloss(self, loss):
        plt.figure()
        plt.plot(loss)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
        
    def train(self, mode='all'):
        if mode=='all':
            ##### train
            self.build_model()
            optimizer = torch.optim.Adam(self.model256.parameters(), lr=self.lr, eps=1e-07)
            print('train model256 now')
            self.model256, loss = self._subtrain(self.model256, optimizer)
            #self.showloss(loss)
            optimizer = torch.optim.Adam(self.model512.parameters(), lr=self.lr, eps=1e-07)
            print('train model512 now')
            self.model512, loss = self._subtrain(self.model512, optimizer)
            #self.showloss(loss)
            optimizer = torch.optim.Adam(self.model1024.parameters(), lr=self.lr, eps=1e-07)
            print('train model1024 now')
            self.model1024, loss = self._subtrain(self.model1024, optimizer)
            #self.showloss(loss)



    def build_model(self,mode='all'):
        if mode=='all':
            self.model256 = MLP(self.inputdim, self.outputdim, self.architectures['m256'][0], self.architectures['m256'][1])
            self.model512 = MLP(self.inputdim, self.outputdim, self.architectures['m512'][0], self.architectures['m512'][1])
            self.model1024 = MLP(self.inputdim, self.outputdim, self.architectures['m1024'][0],self.architectures['m1024'][1])
            self.model1024 = self.model1024.to(device)
            self.model512 = self.model512.to(device)
            self.model256 = self.model256.to(device)
            self.model256.apply(initialize_weight)
            self.model512.apply(initialize_weight)
            self.model1024.apply(initialize_weight)
            
    def predict(self, test_x, mode='all'):
        test_x = torch.from_numpy(test_x).to(device).float()
        if mode == 'all':
            self.model256.eval()
            self.model512.eval()
            self.model1024.eval()
        if mode == 'all':
            pred = (self.model256(test_x) + self.model512(test_x) + self.model1024(test_x)) / 3
        return pred.cpu().detach().numpy()

def test_AE_function(train_x,train_y,test_x,test_y,batch_size=128):
    reproducibility(seed=0)
    train_loader = DataLoader(simdatset(train_x, train_y), batch_size=batch_size, shuffle=True)
    model = AutoEncoder(train_x.shape[1], train_x.shape[2], train_y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    model, loss, reconloss = train(model, train_loader, optimizer, epochs=int(1000/(len(train_x)/batch_size)))
    showloss(loss)
    showloss(reconloss)
    pred, train_sigm = predict(model, test_x)
    #showloss(loss)
    l1, ccc = score(pred,test_y.cpu().detach().numpy())
    #l1, ccc = score(test_pred,test_y)
    return l1, ccc, pred, train_sigm
def test_scaden_function(train_x,train_y,test_x,test_y,batch_size=128):
    reproducibility(seed=0)
    architectures = {'m256': ([256,128,64,32],[0,0,0,0]),
                     'm512': ([512,256,128,64],[0, 0.3, 0.2, 0.1]),
                     'm1024': ([1024, 512, 256, 128],[0, 0.6, 0.3, 0.1])}
    model = scaden(architectures, train_x, train_y, epochs=int(1000/(len(train_x)/batch_size)))
    st = time.time()
    model.train()
    pred = model.predict(test_x)
    ed = time.time()
    l1, ccc = score(pred,test_y)
    return l1, ccc

################## run #######################


if __name__ == "__main__":

    # read command line input parameters
    parser = argparse.ArgumentParser(
        description="CelFEER - Code to perform simulations on generated data."
    )
    parser.add_argument("output_directory", help="the path to the output directory")
    parser.add_argument("num_samples", type=int, help="Number of individuals")
    parser.add_argument("num_tissues", type=int, help="Number of cell types")
    parser.add_argument("num_cpgs", type=int, help="Number of cpg sites")
    parser.add_argument("depth", type=int, help="expected depth of cfdna reads")
    parser.add_argument("beta_depth", type=int, help="expected depth of reference reads")
    parser.add_argument(
        "-m",
        "--max_iterations",
        default=1000,
        type=int,
        help="How long the EM should iterate before stopping, unless convergence criteria is met. Default 1000.",
    )
    parser.add_argument(
        "-u",
        "--unknowns",
        default=0,
        type=int,
        help="Number of tissues in the reference data that should be treated as unknown.",
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

    args = parser.parse_args()
    np.random.seed(args.parallel_job_id)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    Y, X, alpha_true, beta_int = generate_em_replicate(
        args.num_tissues, args.num_cpgs, args.num_samples, args.depth, args.beta_depth, args.unknowns
    )

    data=torch.tensor(X)
    label=torch.tensor(alpha_true)
    # print(true_beta.shape)
    # print(data,label)
    # 5-fold train_test_split
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=5,shuffle=True,random_state=0).split(data,label)
    ### GAUSSIAN NOISE on test data, guassian distribution along each gene
    # print(enumerate(kfold))

    L1 = []
    CCC = []
    for k, (train_indices, test_indices) in enumerate(kfold):
        reproducibility(seed=0)
        # print(train_indices,test_indices)
        train_x = data[train_indices]
        train_y = label[train_indices]
        
        test_x = data[test_indices]
        test_y = label[test_indices]

        # test_beta = beta[test_indices]

        # l1, ccc = test_scaden_function(train_x,train_y,test_x,test_y)
        l1, ccc, pred, sign = test_AE_function(train_x,train_y,test_x,test_y)
        print('l1, ccc:',l1, ccc)
        L1.append(l1)
        CCC.append(ccc)
        print('sign',sign.shape)
        print('true_beta',beta_int.shape)

        with open(args.output_directory + "/" + str(args.parallel_job_id) + "_alpha_est.pkl", "wb") as f:
            pkl.dump(pred, f)

        with open(args.output_directory + "/" + str(args.parallel_job_id) + "_alpha_true.pkl", "wb") as f:
            pkl.dump(alpha_true, f)

        with open(args.output_directory + "/" + str(args.parallel_job_id) + "_beta_est.pkl", "wb") as f:
            pkl.dump(sign, f)

        with open(args.output_directory + "/" + str(args.parallel_job_id) + "_beta_true.pkl", "wb") as f:
            pkl.dump(beta_int, f)

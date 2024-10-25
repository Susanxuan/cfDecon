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


def complex_mix(sample, proportions, unknown_cols, num_tissues):
    """normalizes the read coverage of the input tissue data and multiplies this data with the input proportions
    sets unknown tissues to 0 in the reference and initializes beta

    sample: pandas dataframe of data (samples and reference), assumes there are 3 columns (chrom, start, end)
    before the samples and before the reference
    proportions: desired proportions for each tissue
    unknown_cols: tissues to treat as unknowns
    num_tissues: amount of tissues
    """
    test = sample.iloc[:, 3: (num_tissues) * 5 + 3].values
    train = sample.iloc[:, (num_tissues) * 5 + 3 + 3:].values
    x = np.array(np.split(test, num_tissues, axis=1))
    y = np.array(np.split(train, num_tissues, axis=1))

    tissue_totaldepths = np.sum(x, axis=(1, 2))
    x = (x.T / tissue_totaldepths).T * np.average(tissue_totaldepths)

    x_percents = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

    mix_x_percents = np.dot(proportions, x_percents)

    mix_x = mix_x_percents.reshape(proportions.shape[0], x.shape[1], 5)

    true_beta = y / np.sum(y, axis=2)[:, :, np.newaxis]

    if unknown_cols:
        for col in unknown_cols:
            y[col] = 0

    return np.nan_to_num(mix_x), np.nan_to_num(y), np.nan_to_num(true_beta)

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
    """take in the input cfdna matrices and the reference data and
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

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, channels, output_dim):
        super().__init__()
        self.name = 'ae'
        self.inputdim = input_dim
        self.channels = channels
        self.outputdim = output_dim
        self.encoder = nn.Sequential(nn.Dropout(),
                                     nn.Linear(self.inputdim*self.channels,512),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(512,256),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(256,128),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(128,64),
                                     nn.CELU(),
                                     nn.Linear(64,output_dim),
                                     nn.Softmax()
                                     )
        
        self.decoder = nn.Sequential(nn.Linear(self.outputdim,64,bias=False),
                                     nn.Linear(64,128,bias=False),
                                     nn.Linear(128,256,bias=False),
                                     nn.Linear(256,512,bias=False),
                                     nn.Linear(512,self.inputdim*self.channels,bias=False))
        
                                               
    def encode(self,x):
        encode_temp=x.view(x.size(0), -1)
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
        w04 = torch.mm(w03, w4)
        return F.hardtanh(w04,0,1)

    def forward(self, x):
        sigmatrix = self.sigmatrix()
        z = self.encode(x)
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
    model, loss, reconloss = train(model, train_loader, optimizer, epochs=int(5000/(len(train_x)/batch_size)))
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
        description="CelFEER - Code to perform simulations using WGBS data."
    )
    parser.add_argument("input_path", help="the path to the input file")
    parser.add_argument("output_directory", help="the path to the output directory")
    parser.add_argument("num_samples", type=int, help="Number of cfdna samples")
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
        default=None,
        type=str,
        help="Tissues in the reference data that should be treated as unknown. Give tissue columns separated by " +
             "comma, e.g. 0,3,6. Default is none.",
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
    args = parser.parse_args()
    np.random.seed(args.parallel_job_id)
    unknowns = [int(x) for x in args.unknowns.split(",")] if args.unknowns else None
    num_tissues = args.num_samples

    # make output directory if it does not exist
    if not os.path.exists(args.output_directory) and args.parallel_job_id == 1:
        os.makedirs(args.output_directory)
        print("made " + args.output_directory + "/")
        print()
    else:
        print("writing to " + args.output_directory + "/")

    data_df = pd.read_csv(args.input_path, header=None, delimiter="\t")

    print(f"finished reading {args.input_path}")
    print()

    # If desired, hardcode the proportions here. The current array creates ascending proportions.
    props = np.array([i / sum(range(1, num_tissues + 1)) for i in range(1, num_tissues + 1)]) \
        .reshape(1, num_tissues).repeat(args.num_samples, axis=0)

    # Load file containing proportions
    if args.proportions:
        props = pkl.load(open(args.proportions, 'rb'))

    x, y, true_beta = complex_mix(data_df, props, unknowns, args.num_samples)

    data=torch.tensor(x)
    label=torch.tensor(props)
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
        print(sign.shape)

    # print('data_df',data_df.shape)
    # print('props',props.shape)
    # print('x',x.shape)
    # print('y',y.shape)
    # print('true_beta',true_beta.shape)

    # Run EM with the specified iterations and convergence criteria
    # random_restarts = []

    # for i in range(args.random_restarts):
    #     alpha, beta, ll = em(
    #         x, y, args.max_iterations, args.convergence
    #     )
    #     random_restarts.append((ll, alpha, beta))

    # ll_max, alpha_max, beta_max = max(random_restarts)  # pick best random restart per replicate
    # print('alpha_max',alpha_max.shape)
        with open(args.output_directory + "/" + str(k) + "_alpha_est.pkl", "wb") as f:
            pkl.dump(pred, f)

        with open(args.output_directory + "/" + str(k) + "_alpha_true.pkl", "wb") as f:
            pkl.dump(props, f)

        with open(args.output_directory + "/" + str(k) + "_beta_est.pkl", "wb") as f:
            pkl.dump(sign, f)

        # with open(args.output_directory + "/" + str(k) + "_beta_true.pkl", "wb") as f:
        #     pkl.dump(test_beta, f)
    print('all:',sum(L1)/len(L1),sum(CCC)/len(CCC))

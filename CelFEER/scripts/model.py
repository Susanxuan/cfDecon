import torch
import random
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


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

class ChannelDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ChannelDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 64, bias=False),
            nn.Linear(64, 128, bias=False),
            nn.Linear(128, 256, bias=False),
            nn.Linear(256, 512, bias=False),
            nn.Linear(512, output_dim, bias=False)
        )

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
        return F.relu(w04)

    def forward(self, z):
        sigmatrix = self.sigmatrix()
        x_recon = torch.mm(z,sigmatrix)
        return x_recon, sigmatrix


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
        stacked_tensor = torch.stack(encoded_outputs, dim=2)
        compressed_tensor = self.compressor(stacked_tensor.view(stacked_tensor.size(0), -1))
        return stacked_tensor, compressed_tensor
    
class MultiChannelDecoder(nn.Module):
    def __init__(self, input_dim, channels, output_dim):
        super(MultiChannelDecoder, self).__init__()
        self.channels = channels
        self.output_dim = output_dim
        self.decoders = nn.ModuleList([ChannelDecoder(output_dim, input_dim) for _ in range(channels)])
        self.decompressor = nn.Linear(1, channels, bias=False)

    def forward(self, z):
        decoded_outputs = []
        signatrixs = []
        # decompressed_latent = self.decompressor(z.unsqueeze(-1))
        decompressed_latent = z.unsqueeze(-1).repeat(1,1,self.channels)
        for i in range(self.channels):
            # decoded_output,signatrix = self.decoders[i](z)
            decoded_output,signatrix = self.decoders[i](decompressed_latent[:,:,i])
            decoded_outputs.append(decoded_output)
            signatrixs.append(signatrix)
        stacked_recon = torch.stack(decoded_outputs, dim=2)
        stacked_sig = torch.stack(signatrixs, dim=2)
        return stacked_recon,stacked_sig
    

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, channels, output_dim):
        super().__init__()
        self.name = 'ae'
        self.inputdim = input_dim
        self.channels = channels
        self.outputdim = output_dim
        self.encoder = MultiChannelEncoder(self.inputdim, self.channels, self.outputdim)
        self.decoder = MultiChannelDecoder(self.inputdim, self.channels, self.outputdim)
                                               
    def encode(self,x):
        return self.encoder(x)
    
    def decode(self,z):
        return self.decoder(z)

    def forward(self, x):
        stacked_latent, compressed_latent = self.encode(x)
        stacked_recon, stacked_sig = self.decoder(compressed_latent)
        return stacked_recon, compressed_latent, stacked_sig
import torch
import random
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
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


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16, dtype=torch.float):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False, dtype=dtype),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x.T)
        y = self.fc(y.T)
        # a = y.cpu().detach().numpy()
        a = y.expand_as(x)
        return x * y.expand_as(x)


class new_combine_encoder(nn.Module):
    def __init__(self, input_dim, output_dim, r=[32, 24, 16, 12], use_layernorm=False):
        super(new_combine_encoder, self).__init__()
        self.r = r
        self.use_layernorm = use_layernorm  # setting for LayerNorm
        self.weights = nn.ParameterList()  # setting for weight saving
        self.weights.append(nn.Parameter(torch.Tensor(5, input_dim, self.r[0])))  # first weight

        for i in range(1, len(self.r)):
            self.weights.append(nn.Parameter(torch.Tensor(5, self.r[i - 1], self.r[i])))  # dynamic weight addition

        self.weights.append(nn.Parameter(torch.Tensor(5, self.r[-1], output_dim)))  # last weight

        self.output_dim = output_dim
        self.leakyrelu_layers = nn.ModuleList([nn.LeakyReLU() for _ in range(len(self.r) + 1)])  # LeakyReLU

        if self.use_layernorm:
            self.layernorms = nn.ModuleList([nn.LayerNorm(dim) for dim in self.r])  # LayerNorm 

        self.gnet = nn.Sequential(
            nn.Linear(5, 5, bias=False),
            nn.LeakyReLU(),
            nn.Linear(5, 5, bias=False)
        )

        self.compressor = nn.Sequential(
            nn.Linear(output_dim * 5, output_dim),
            nn.Softmax(dim=1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        def uniform_init(w):
            fan_in = w.size(1)
            bound = 1. / torch.sqrt(
                torch.tensor(w.size(0), dtype=torch.float32) + torch.sqrt(torch.tensor(fan_in, dtype=torch.float32)))
            nn.init.uniform_(w, -bound, bound)

        for w in self.weights:
            uniform_init(w)

    def forward(self, x):
        x = x.permute(2, 0, 1)

        for i in range(len(self.weights) - 1):
            x = torch.matmul(x, self.weights[i])
            x = self.leakyrelu_layers[i](x)
            if self.use_layernorm:
                x = self.layernorms[i](x)

        A = torch.matmul(x, self.weights[-1])

        out = self.gnet(A.permute(1, 2, 0))
        compressed_tensor = self.compressor(out.reshape(out.size(0), out.size(1) * out.size(2)))
        return out, compressed_tensor


class new_combine_decoder(nn.Module):
    def __init__(self, input_dim, output_dim, r=[12, 16, 24, 32]):
        super(new_combine_decoder, self).__init__()
        self.r = r
        self.weights = nn.ParameterList()  # save all weights
        self.weights.append(nn.Parameter(torch.Tensor(5, output_dim, self.r[0]))) 

        for i in range(1, len(self.r)):
            self.weights.append(nn.Parameter(torch.Tensor(5, self.r[i - 1], self.r[i])))  

        self.weights.append(nn.Parameter(torch.Tensor(5, self.r[-1], input_dim)))  

        self.output_dim = output_dim
        self.leakyrelu1 = nn.LeakyReLU()
        # self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=output_dim)
        # self.attention1 = SE_Block(ch_in=5, reduction=1)
        # self.attention2 = SE_Block(ch_in=5, reduction=1)
        self.reset_parameters()

    def reset_parameters(self):
        def uniform_init(w):
            fan_in = w.size(1)
            bound = 1. / torch.sqrt(
                torch.tensor(w.size(0), dtype=torch.float32) + torch.sqrt(torch.tensor(fan_in, dtype=torch.float32)))
            nn.init.uniform_(w, -bound, bound)

        for w in self.weights:
            uniform_init(w)

    def sigmatrix(self):
        # 动态构建sigmatrix
        sigmatrix = self.weights[0]
        for i in range(1, len(self.weights)):
            sigmatrix = torch.matmul(sigmatrix, self.weights[i])
        return self.leakyrelu1(sigmatrix)  # one leakyrelu self.leakyrelu1()

    def forward(self, z):
        z = z.unsqueeze(2).expand(-1, -1, 5).permute(2, 0, 1)  # 5 7 9
        sigmatrix = self.sigmatrix()  # 5 9 134
        # sigmatrix = self.attention1(sigmatrix.permute(1, 2, 0))
        out = torch.matmul(z, sigmatrix)  # 5 7 134
        out = out.permute(1, 2, 0)
        sigmatrix = sigmatrix.permute(1, 2, 0)

        # out, _ = self.attention(out, out, out)
        # sigmatrix = self.attention1(sigmatrix)
        # out = self.attention2(out)
        return out, sigmatrix


class ChannelEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ChannelEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_dim, 1024),
            nn.CELU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
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
        w01 = torch.mm(w0, w1)
        w02 = torch.mm(w01, w2)
        w03 = torch.mm(w02, w3)
        w04 = torch.mm(w03, w4)
        return F.relu(w04)

    def forward(self, z):
        sigmatrix = self.sigmatrix()
        x_recon = torch.mm(z, sigmatrix)
        return x_recon, sigmatrix





class cfDecon(nn.Module):
    def __init__(self, input_dim, channels, output_dim, r1, r2, use_layernorm):
        super().__init__()
        self.name = 'ae'
        self.inputdim = input_dim
        self.channels = channels
        self.outputdim = output_dim
        # self.encoder = MultiChannelEncoder(self.inputdim, self.channels, self.outputdim)
        self.encoder = new_combine_encoder(self.inputdim, self.outputdim, r1,
                                           use_layernorm=(use_layernorm == str(True)))
        # self.decoder = MultiChannelDecoder(self.inputdim, self.channels, self.outputdim)
        self.decoder = new_combine_decoder(self.inputdim, self.outputdim, r2)  # [32, 64, 128, 256]

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        stacked_latent, compressed_latent = self.encode(x)
        stacked_recon, stacked_sig = self.decoder(compressed_latent)
        # stacked_recon, stacked_sig = self.decoder(stacked_latent)
        return stacked_recon, compressed_latent, stacked_sig


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
            encoded_output = self.encoders[i](x[:, :, i])
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

    def signnorm(self, sig):
        sum_last_dim = torch.sum(sig, axis=2, keepdims=True)
        norm_sigmatrix = sig / sum_last_dim
        return norm_sigmatrix

    def forward(self, z):
        decoded_outputs = []
        signatrixs = []
        # decompressed_latent = self.decompressor(z.unsqueeze(-1))
        decompressed_latent = z.unsqueeze(-1).repeat(1, 1, self.channels)
        for i in range(self.channels):
            # decoded_output,signatrix = self.decoders[i](z)
            decoded_output, signatrix = self.decoders[i](decompressed_latent[:, :, i])
            decoded_outputs.append(decoded_output)
            signatrixs.append(signatrix)
        stacked_recon = torch.stack(decoded_outputs, dim=2)
        stacked_sig = torch.stack(signatrixs, dim=2)
        norm_stacked_sig = self.signnorm(stacked_sig)
        return stacked_recon, norm_stacked_sig

class baseline(nn.Module):
    def __init__(self, input_dim, channels, output_dim):
        super().__init__()
        self.name = 'ae'
        self.inputdim = input_dim
        self.channels = channels
        self.outputdim = output_dim
        self.encoder = MultiChannelEncoder(self.inputdim, self.channels, self.outputdim)
        self.decoder = MultiChannelDecoder(self.inputdim, self.channels, self.outputdim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        stacked_latent, compressed_latent = self.encode(x)
        stacked_recon, stacked_sig = self.decoder(compressed_latent)
        # stacked_recon, stacked_sig = self.decoder(stacked_latent)
        return stacked_recon, compressed_latent, stacked_sig

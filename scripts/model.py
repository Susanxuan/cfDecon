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
        self.use_layernorm = use_layernorm  # 控制是否使用 LayerNorm 的开关
        self.weights = nn.ParameterList()  # 用于存储所有权重参数
        self.weights.append(nn.Parameter(torch.Tensor(5, input_dim, self.r[0])))  # 第一个权重参数

        for i in range(1, len(self.r)):
            self.weights.append(nn.Parameter(torch.Tensor(5, self.r[i-1], self.r[i])))  # 动态添加权重参数

        self.weights.append(nn.Parameter(torch.Tensor(5, self.r[-1], output_dim)))  # 最后一个权重参数

        self.output_dim = output_dim
        self.leakyrelu_layers = nn.ModuleList([nn.LeakyReLU() for _ in range(len(self.r) + 1)])  # 动态创建LeakyReLU层

        if self.use_layernorm:
            self.layernorms = nn.ModuleList([nn.LayerNorm(dim) for dim in self.r])  # 动态创建LayerNorm层

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
            bound = 1. / torch.sqrt(torch.tensor(w.size(0), dtype=torch.float32) + torch.sqrt(torch.tensor(fan_in, dtype=torch.float32)))
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
        self.weights = nn.ParameterList()  # 用于存储所有权重参数
        self.weights.append(nn.Parameter(torch.Tensor(5, output_dim, self.r[0])))  # 第一个权重参数

        for i in range(1, len(self.r)):
            self.weights.append(nn.Parameter(torch.Tensor(5, self.r[i-1], self.r[i])))  # 动态添加权重参数

        self.weights.append(nn.Parameter(torch.Tensor(5, self.r[-1], input_dim)))  # 最后一个权重参数

        self.output_dim = output_dim
        self.leakyrelu1 = nn.LeakyReLU()
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=output_dim)
        self.attention1 = SE_Block(ch_in=5, reduction=1)
        self.attention2 = SE_Block(ch_in=5, reduction=1)
        self.reset_parameters()

    def reset_parameters(self):
        def uniform_init(w):
            fan_in = w.size(1)
            bound = 1. / torch.sqrt(torch.tensor(w.size(0), dtype=torch.float32) + torch.sqrt(torch.tensor(fan_in, dtype=torch.float32)))
            nn.init.uniform_(w, -bound, bound)

        for w in self.weights:
            uniform_init(w)

    def sigmatrix(self):
        # 动态构建sigmatrix
        sigmatrix = self.weights[0]
        for i in range(1, len(self.weights)):
            sigmatrix = torch.matmul(sigmatrix, self.weights[i])
        return self.leakyrelu1(sigmatrix)

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
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(ChannelEncoder, self).__init__()
        layers = []
        layer_dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            layers.append(nn.Dropout())
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(nn.CELU())

        layers.append(nn.Linear(layer_dims[-1], output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class ChannelDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(ChannelDecoder, self).__init__()
        layers = []
        layer_dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=False))

        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1], bias=False))
        self.decoder = nn.Sequential(*layers)

    def sigmatrix(self):
        weights = [self.decoder[i].weight.T for i in range(len(self.decoder))]
        sigmatrix = weights[0]
        for i in range(1, len(weights)):
            sigmatrix = torch.mm(sigmatrix, weights[i])
        return F.relu(sigmatrix)

    def forward(self, z):
        sigmatrix = self.sigmatrix()
        x_recon = torch.mm(z, sigmatrix)
        return x_recon, sigmatrix


class MultiChannelEncoder(nn.Module):
    def __init__(self, input_dim, channels, output_dim):
        super(MultiChannelEncoder, self).__init__()
        self.channels = channels
        self.encoders = nn.ModuleList(
            [ChannelEncoder(input_dim, output_dim, hidden_dims=[512, 256, 128, 64]) for _ in range(channels)])
        self.compressor = nn.Sequential(
            nn.Linear(output_dim * channels, output_dim),
            # nn.Hardtanh(0,1)
            nn.Softmax(dim=1)
            # nn.ReLU()
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
        self.decoders = nn.ModuleList(
            [ChannelDecoder(output_dim, input_dim, hidden_dims=[64, 128, 256, 512]) for _ in range(channels)])
        self.decompressor = nn.Linear(1, channels, bias=False)
        self.attention1 = SE_Block(ch_in=5, reduction=1)
        self.attention2 = SE_Block(ch_in=5, reduction=1)

    def forward(self, z):
        decoded_outputs = []
        signatrixs = []
        decompressed_latent = z.unsqueeze(-1).repeat(1, 1, self.channels)
        for i in range(self.channels):
            decoded_output, signatrix = self.decoders[i](decompressed_latent[:, :, i])
            decoded_outputs.append(decoded_output)
            signatrixs.append(signatrix)
        stacked_recon = torch.stack(decoded_outputs, dim=2)
        # stacked_recon = self.attention1(stacked_recon)
        stacked_sig = torch.stack(signatrixs, dim=2)
        # stacked_sig = self.attention2(stacked_sig)
        return stacked_recon, stacked_sig


class DTNN(nn.Module):
    def __init__(self, input_dim, channels, output_dim, r1, r2, use_layernorm):
        super().__init__()
        self.name = 'ae'
        self.inputdim = input_dim
        self.channels = channels
        self.outputdim = output_dim
        #self.encoder = MultiChannelEncoder(self.inputdim, self.channels, self.outputdim)
        self.encoder = new_combine_encoder(self.inputdim, self.outputdim, r1, use_layernorm=use_layernorm)
        #self.decoder = MultiChannelDecoder(self.inputdim, self.channels, self.outputdim)
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


class dnaTape(nn.Module):
    def __init__(self, input_dim, channels, output_dim, r1, r2, use_layernorm):
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

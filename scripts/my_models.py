import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BertModel, BertTokenizer


def noised(inputs, noise_rate):
    noise = inputs.data.new(inputs.size()).normal_(0, 1)
    noised_inputs = inputs + noise * noise_rate 
    
    return noised_inputs


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    
    return mu + eps * std


def activation_fn(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError


class PoseEmbLayer(nn.Module):
    def __init__(
        self,
        layer_dims: list = [240, 1024, 512],
        act: str = 'relu',
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i+2 < len(layer_dims):
                layers.append(activation_fn(act))
                layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PoseGenerator(nn.Module):
    def __init__(
        self,
        layer_dims: list = [512, 1024, 240],
        act: str = 'relu',
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i + 1 < len(layer_dims) - 1:
                layers.append(activation_fn(act))
                layers.append(nn.Dropout(dropout))            
            else:
                layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.layers(z)


class LatentGenerator(nn.Module):
    def __init__(
        self,
        layer_dims: list = [768, 1024, 512, 64],
        act: str = 'relu',
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if i+2 < len(layer_dims):
                layers.append(activation_fn(act))
                layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)


class SignPoseVAE(nn.Module):
    def __init__(
        self,
        enc_layer_dims: list = [240, 1024, 512],
        dec_layer_dims: list = [512, 1024, 240],
        z_dim: int = 64,
        act: str = 'relu',
        noise_rate: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.noise_rate = noise_rate

        enc_layer_dims = enc_layer_dims + [z_dim]
        dec_layer_dims = [z_dim] + dec_layer_dims
        
        self.encoder = PoseEmbLayer(enc_layer_dims, act, dropout)
        
        self.decoder = PoseGenerator(dec_layer_dims, act, dropout)

        self.to_mu = nn.Linear(z_dim, z_dim)
        self.to_logvar = nn.Linear(z_dim, z_dim)
        
    def forward(self, x):
        if self.training and self.noise_rate > 0.0:
            x = noised(x, self.noise_rate)

        enc_outputs = self.encoder(x)

        mu = self.to_mu(enc_outputs)
        logvar = self.to_logvar(enc_outputs)
        
        if self.training:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        
        recon_x = self.decoder(z)

        return {
            'recon_x': recon_x,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    def predict(self, z):
        return self.decoder(z)
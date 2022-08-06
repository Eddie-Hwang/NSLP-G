import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BertModel, BertTokenizer


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
                layers.append(nn.LayerNorm(out_dim))
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
                layers.append(nn.LayerNorm(out_dim))
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
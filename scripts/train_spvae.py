import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI
from torch.utils.data import DataLoader, Dataset

from loss import SignPoseLossForVAE
from models.my_models import SignPoseVAE
from render import draw_skeleton
from spvae_module import Phoenix14TDatasetModule, SignPoseVAEModule
from utils import *


class SignPoseGenerator(SignPoseVAEModule):
    def __init__(
        self,
        verbose: bool = False,
        enc_layer_dims: list = [240, 1024, 512],
        dec_layer_dims: list = [512, 1024, 240],
        z_dim: int = 64,
        c_dim: int = 9,
        act: str = 'relu',
        loss_type: str = 'bce',
        p_weight: float = 0.1,
        l_weight: float = 0.5,
        h_weight: float = 1.5,
        noise_rate: float = 0.01,
        kld_weight: float = 1e-4,
        dropout: float = 0.1,
        lr: float = 1e-4,
    ):        
        model = SignPoseVAE(
            enc_layer_dims=enc_layer_dims,
            dec_layer_dims=dec_layer_dims,
            z_dim=z_dim,
            act=act,
            noise_rate=noise_rate,
            dropout=dropout
        )
        
        loss_fn = SignPoseLossForVAE(
            loss_type=loss_type, 
            kld_weight=kld_weight
        )
        
        super().__init__(model=model, loss_fn=loss_fn, lr=lr)


def _callbacks():
    early_stopping_callback = EarlyStopping(
        monitor='valid_total_loss', 
        patience=30, 
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        filename=f'best',
        save_top_k=1,
        monitor='valid_total_loss',
        mode='min'
    )

    return [early_stopping_callback, checkpoint_callback]


def cli_main():
    cli = LightningCLI(
        SignPoseGenerator,
        Phoenix14TDatasetModule,
        seed_everything_default=42, 
        save_config_overwrite=True,
        trainer_defaults={
            'logger': pl_loggers.TensorBoardLogger("spvae_logs"),
            'max_epochs': 1,
            'fast_dev_run': True,
            'gpus': [1],
            'callbacks': _callbacks(),
            'enable_checkpointing': True,
            'gradient_clip_val': 0.5,
            'precision': 32
        },
        run=False
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # if cli.trainer.fast_dev_run == False:
    #     cli.trainer.test(ckpt_path='best')
    

if __name__=='__main__':
    cli_main()

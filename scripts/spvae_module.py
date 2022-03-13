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

from dataset import Phoenix14Dataset
from loss import SignPoseLossForVAE
from models.my_models import SignPoseVAE
from render import draw_skeleton
from utils import *


class Phoenix14TDatasetModule(pl.LightningDataModule):
    def __init__(
        self,
        train_annotation_path: str = './data/phoenix14t.pose.train',
        valid_annotation_path: str = './data/phoenix14t.pose.dev',
        test_annotation_path: str = './data/phoenix14t.pose.test',
        train_processed_path: str = './data/phoenix14t.lcvae.train',
        valid_processed_path: str = './data/phoenix14t.lcvae.dev',
        test_processed_path: str = './data/phoenix14t.lcvae.test',
        batch_size: int = 2,
        num_workers: int = 0,
        num_process: int = 20,
        sampling_rate: float = 0.1,
    ):
        super().__init__()

        self.train_annotation_path = train_annotation_path  
        self.valid_annotation_path = valid_annotation_path
        self.test_annotation_path = test_annotation_path

        self.train_processed_path = train_processed_path
        self.valid_processed_path = valid_processed_path
        self.test_processed_path = test_processed_path

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_process = num_process
        self.sampling_rate = sampling_rate

    def collate_fn(self, batch):
        condition_batch = []
        skeleton_batch = []
        for data in batch:
            condition_batch.append(data['condition'])
            skeleton_batch.append(data['skeleton'])

        return {
            'condition': condition_batch,
            'skeleton': torch.Tensor(skeleton_batch)
        }

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_data = Phoenix14Dataset(
                annotation_path=self.train_annotation_path,
                processed_data_path=self.train_processed_path,
                n_proc=self.num_process,
                sampling_rate=self.sampling_rate,
                mode='vae'
            )
            self.valid_data = Phoenix14Dataset(
                annotation_path=self.valid_annotation_path,
                processed_data_path=self.valid_processed_path,
                n_proc=self.num_process,
                sampling_rate=self.sampling_rate,
                mode='vae'
            )
        else:
            self.test_data = Phoenix14Dataset(
                annotation_path=self.test_annotation_path,
                processed_data_path=self.test_processed_path,
                n_proc=self.num_process,
                sampling_rate=self.sampling_rate,
                mode='vae'
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )


class SignPoseVAEModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module = None,
        loss_fn: nn.Module = None,
        lr: float = 1e-4
    ):
        super().__init__()
        
        assert model != None, 'Model must be defined.'
        assert loss_fn != None, 'Loss function must be defined.'

        self.model = model
        self.loss_fn = loss_fn
        
        self.lr = lr

    def common_step(self, batch, stage):
        x = batch['skeleton']

        bsz, _ = x.size()
                
        outputs = self.model(x)

        recon_x = outputs['recon_x']
        mu = outputs['mu']
        logvar = outputs['logvar']
        z = outputs['z']
        
        losses = self.loss_fn(recon_x, x, mu, logvar)

        total_loss = losses['total_loss']
        dist_loss = losses['dist_loss']
        kld_loss = losses['kld_loss']
        
        # logging
        self.log(f'{stage}/total_loss', total_loss, batch_size=bsz)
        self.log(f'{stage}/dist_loss', dist_loss, batch_size=bsz)
        self.log(f'{stage}/kld_loss', kld_loss, batch_size=bsz)

        return {
            'loss': total_loss,
            'x': x.detach().cpu(),
            'recon_x': recon_x.detach().cpu(),
            'z': z.detach().cpu(),
        }
   
    def common_epoch_end(self, outputs: dict, stage: str):
        if stage != 'test':
            n_sample = 1
        else:
            n_sample = 5
        outputs = random.sample(outputs, n_sample)

        # logging latent space
        latent_space = []
        recon_images = []
        target_images = []
        for output in outputs:
            latent_space.append(output['z']) # [b, latent_size]
            recon_images.append(output['recon_x'])
            target_images.append(output['x'])
        
        latent_space = torch.stack([l for latent in latent_space for l in latent])
        recon_images = torch.stack([img for image in recon_images for img in image])
        target_images = torch.stack([img for image in target_images for img in image])
        
        # post processing
        target_images = relocate_landmark(target_images.numpy())
        recon_images = relocate_landmark(recon_images.numpy())
        
        # get randered images
        recon_images = [draw_skeleton(img, np.zeros((256,256,3), np.uint8) + 255) for img in recon_images]
        recon_images = rearrange(torch.tensor(recon_images), 'b h w c -> b c h w')
        target_images = [draw_skeleton(img, np.zeros((256,256,3), np.uint8) + 255) for img in target_images]
        target_images = rearrange(torch.tensor(target_images), 'b h w c -> b c h w')

        assert len(latent_space) == len(recon_images)

        x_recon_grid = torchvision.utils.make_grid(recon_images[:10], nrow=5)
        x_grid = torchvision.utils.make_grid(target_images[:10], nrow=5)

        self.logger.experiment.add_image(f'{stage}/generated_sign', x_recon_grid, self.global_step)
        self.logger.experiment.add_image(f'{stage}/reference_sign', x_grid, self.global_step)
        
        if stage == 'test':
            self.logger.experiment.add_embedding(latent_space, label_img=target_images)

    def training_step(self, batch: dict, batch_idx: int):
        return self.common_step(batch, 'tr')

    def validation_step(self, batch: dict, batch_idx: int):
        return self.common_step(batch, 'val')

    def validation_epoch_end(self, outputs: dict):
        self.common_epoch_end(outputs, 'val')

    def test_step(self, batch: dict, batch_idx: int):
        return self.common_step(batch, 'tst')

    def test_epoch_end(self, outputs: dict):
        self.common_epoch_end(outputs, 'tst')

    def configure_optimizers(self): 
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
    
        return {
            'optimizer': optimizer,
        }


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
        SignPoseVAEModule,
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

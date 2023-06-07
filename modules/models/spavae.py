from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from modules.noise_scheduler import noise_scheduler_dict
from einops_exts import check_shape
from modules.helpers import create_mask, add_noise, instantiate_from_config
from modules.normalize import scaling_keypoints, center_keypoints
from modules.constant import return_scale_resoution
from modules.models.mlp import MLP, ResidualMLP


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    
    return mu + eps * std


class SpatialVAE(pl.LightningModule):
    def __init__(
        self,
        num_joints=50,
        num_feats=3,
        latent_dim=512,
        noise=None,
        initial_noise_std=0.5, 
        final_noise_std=0.1, 
        sharpness_factor=1.,
        base_learning_rate=0.0001,
        monitor=None,
        scheduler_config=None,
        ckpt_path=None,
        ignore_keys=[],
        kl_weight=0.000001,
        loss_type="mse",
        **kwargs
    ):
        super().__init__()

        self.kl_weight = kl_weight
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.learning_rate = base_learning_rate
        self.monitor = monitor
        self.scheduler_config = scheduler_config
        self.num_joints = num_joints
        self.num_feats = num_feats

        try:
            noise_scheduler = noise_scheduler_dict[noise]
            self.noise_scheduler = partial(noise_scheduler, initial_noise_std=initial_noise_std, final_noise_std=final_noise_std, sharpness_factor=sharpness_factor)
        except KeyError:
            print("No such a noise scheduler implemented. Choose from [linear, exp, cosine_annealing, constant]")
        
        pose_dims = num_joints * num_feats
        self.encoder = ResidualMLP(input_size=pose_dims, output_size=latent_dim)
        self.decoder = MLP(input_size=latent_dim, output_size=pose_dims)

        self.to_mu = nn.Linear(latent_dim, latent_dim)
        self.to_logvar = nn.Linear(latent_dim, latent_dim)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, x, mask, noise_std=0.):
        check_shape(x, 'b f v c', v=self.num_joints, c=self.num_feats)
        x = rearrange(x, 'b f v c -> b f (v c)')
        
        mu, logvar = self.encode(add_noise(x, noise_std))        
        z = reparameterize(mu, logvar)

        recon_x = self.generate(z, mask)

        return recon_x, mu, logvar

    def encode(self, x):
        outs = self.encoder(x)
        mu = self.to_mu(outs)
        logvar = self.to_logvar(outs)
        return mu, logvar
    
    def generate(self, z, mask):
        outputs = self.decoder(z)
        outputs[~mask] = 0
        outputs = rearrange(outputs, "b f (v c) -> b f v c", v=self.num_joints, c=self.num_feats)
        return outputs    

    def get_input(self, batch):
        text = batch["text"]
        gloss = batch["gloss"]
        keypoints = batch["keypoints"]
        frame_lengths = batch["frame_lengths"]
        mask = create_mask(frame_lengths, device=self.device)
        
        return text, gloss, keypoints, mask

    def share_step(self, inputs, mask, split="train"):
        noise_std = self.noise_scheduler(total_steps=self.trainer.estimated_stepping_batches, current_step=self.global_step)
        reconstructions, mu, logvar = self(inputs, mask, noise_std)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = F.mse_loss(reconstructions[mask].contiguous(), inputs[mask].contiguous())

        loss = recon_loss + self.kl_weight * kl_loss

        log_dict = {
            f"{split}/kl_loss": kl_loss.detach().mean(),
            f"{split}/recon_loss": recon_loss.detach().mean()
        }

        return loss, log_dict
    
    def training_step(self, batch, batch_idx):
        _, _, inputs, mask = self.get_input(batch)
        loss, log_dict = self.share_step(inputs, mask, split="train")
        
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=inputs.shape[0])
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=inputs.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        _, _, inputs, mask = self.get_input(batch)
        _, log_dict = self.share_step(inputs, mask, split="valid")

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=inputs.shape[0])

    def configure_optimizers(self):
        lr = self.learning_rate

        optim = torch.optim.Adam(
            list(self.encoder.parameters())+
            list(self.decoder.parameters())+
            list(self.to_mu.parameters())+
            list(self.to_logvar.parameters()),
            lr=lr, betas=(0.5, 0.9))
        
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(optim, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            
            return [optim], scheduler
        return optim

    def log_keypoints(self, batch, **kwargs):
        log = dict()

        text, _, inputs, mask = self.get_input(batch)

        mu, _ = self.encode(rearrange(inputs, 'b f v c -> b f (v c)'))
        z = mu

        reference = inputs

        reconstructions = self.generate(z, mask)

        reference, reconstructions = list(map(partial(scaling_keypoints, 
                            width=return_scale_resoution()[0], height=return_scale_resoution()[1]), [reference, reconstructions]))
        
        reference, reconstructions = list(map(partial(center_keypoints, joint_idx=1), [reference, reconstructions]))

        log["reference"] = reference
        log["generated"] = reconstructions
        log["text"] = text

        return log 
import torch
import torch.nn as nn
from torch.nn import functional as F


class SignPoseLossForVAE(nn.Module):
    def __init__(
        self,
        loss_type: str = 'bce',
        kld_weight: float = 1e-4
    ):
        super().__init__()

        self.kld_weight = kld_weight
        self.loss_type = loss_type

        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'bce':
            self.criterion = nn.BCELoss(reduction='sum')
            # self.criterion = nn.BCELoss()
        else:
            raise NotImplementedError

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, recon_x, x, mu=None, logvar=None):
        bsz = recon_x.size(0)

        # distance loss
        dist_loss = self.criterion(recon_x, x)
        
        # kld loss
        if self.kld_weight > 0.0:
            kld_loss = self.kld_loss(mu, logvar)
        else:
            kld_loss = 0.0
        
        # total loss
        total_loss = dist_loss + kld_loss * self.kld_weight

        return {
            'total_loss': total_loss,
            'dist_loss': dist_loss,
            'kld_loss': kld_loss
        }
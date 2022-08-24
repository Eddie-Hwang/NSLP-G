import math
import os
import random
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data import load_data
from layers import PoseEmbLayer, PoseGenerator
from render import save_sign_video, save_sign_video_batch
from utils import noised, postprocess


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    
    return mu + eps * std


class SpatialVAE(nn.Module):
    def __init__(
        self,
        layer_dims,
        latent_dim,
        act,
        noise_rate,
        dropout,
        kl_weight,
        anneal_rate,
        loss_type,
        **kwargs
    ):
        super().__init__()

        enc_layer_dims = layer_dims + [latent_dim]
        dec_layer_dims = [latent_dim] + list(reversed(layer_dims))

        encoder = PoseEmbLayer(enc_layer_dims, act, dropout)
        decoder = PoseGenerator(dec_layer_dims, act, dropout)

        to_mu = nn.Linear(latent_dim, latent_dim)
        to_logvar = nn.Linear(latent_dim, latent_dim)

        self.encoder = encoder
        self.decoder = decoder
        self.to_mu = to_mu
        self.to_logvar = to_logvar

        self.noise_rate = noise_rate
        self.kl_weight = kl_weight
        self.anneal_rate = anneal_rate
        self.latent_dim = latent_dim

        self.loss_type = loss_type

    def forward(self, x, global_step):
        if self.training and self.noise_rate > 0.0:
            noised_x = noised(x, self.noise_rate)
        else:
            noised_x = x

        enc_outs = self.encoder(noised_x)

        mu = self.to_mu(enc_outs)
        logvar = self.to_logvar(enc_outs)
        
        if self.training:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        
        recon_x = self.decoder(z)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if self.loss_type != 'mse':
            recon_loss = F.binary_cross_entropy(recon_x, x)
        else:
            recon_loss = F.mse_loss(recon_x, x)

        if global_step != 0:
            self.kl_weight = min(self.kl_weight * math.exp(self.anneal_rate * global_step), 0.1)

        loss = recon_loss + self.kl_weight * kl_loss

        return loss, recon_loss, kl_loss, recon_x

    def generate(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def encode(self, x):
        enc_outs = self.encoder(x)
        return self.to_mu(enc_outs)

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('sp_vqvae')
        parser.add_argument('--layer_dims', nargs = '+', type = int, default = [240, 1024, 2048, 512])
        parser.add_argument('--noise_rate', type = float, default = 0.001)
        parser.add_argument('--loss_type', type = str, default = 'bce')
        parser.add_argument('--kl_weight', type = float, default = 1e-8)
        parser.add_argument('--anneal_rate', type = float, default = 1e-10)
        parser.add_argument('--latent_dim', type = int, default = 64)
        parser.add_argument('--act', type = str, default = 'relu')
        parser.add_argument('--dropout', type = float, default = 0.)
        
        return parent_parser


class SpatialVAEModule(pl.LightningModule):
    def __init__(
        self, 
        dataset_type,
        seq_len,
        num_save,
        train_path,
        valid_path,
        test_path,
        batch_size,
        num_workers,
        lr,
        **kwargs
    ):
        super().__init__()
        
        self.save_hyperparameters()

        # define model
        self.model = SpatialVAE(**kwargs)

        # dataset related parameters
        self.dataset_type = dataset_type
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.seq_len = seq_len

        # Training related paramters
        self.num_save = num_save
        self.batch_size = batch_size
        self.num_worker = num_workers
        self.lr = lr

    def _common_step(self, batch, stage):
        id = batch['id']
        text = batch['text']
        joints = batch['joint_feats']

        joints = rearrange(joints, 'b t v c -> b t (v c)')

        loss, recon_loss, kl_loss, _ = self.model(joints, self.global_step)

        self.log(f'{stage}/loss', loss, batch_size = self.batch_size)
        self.log(f'{stage}/recon_loss', recon_loss, batch_size = self.batch_size)
        self.log(f'{stage}/kl_loss', kl_loss, batch_size = self.batch_size)

        if stage == 'tr':
            return loss
        
        # full encode and decode
        if stage == 'val':
            full_dataset = self.validset_full
        else:
            full_dataset = self.testset_full
        
        _vf_data = random.choice(full_dataset)
        
        _joint = _vf_data['joint_feats']
        _text = _vf_data['text']
        _id = _vf_data['id']
        
        _joint = rearrange(_joint, 't v c -> t (v c)')
        _z = self.model.encode(_joint.to(self.device))
        _decoded = self.model.generate(_z)

        _joint = rearrange(_joint, 't (v c) -> t v c', c = 2)
        _decoded = rearrange(_decoded, 't (v c) -> t v c', c = 2)
        
        return {
            'loss': loss,
            'text': text,
            'id': id,
            '_joint': _joint.cpu().detach(),
            '_decoded': _decoded.cpu().detach(),
            '_text': _text,
            '_id': _id
        }

    def setup(self, stage):        
        # load dataset
        self.trainset, self.validset, self.testset = \
            load_data(
                self.dataset_type, 
                self.train_path, 
                self.valid_path, 
                self.test_path, 
                seq_len = self.seq_len, 
                min_seq_len = self.seq_len
            )
        _, self.validset_full, self.testset_full = \
            load_data(
                self.dataset_type, 
                self.train_path, 
                self.valid_path, 
                self.test_path, 
                min_seq_len = self.seq_len
            )

        if self.dataset_type == 'how2sign':
            self.S = 3
        else:
            self.S = 1.5
        
        print(f'[INFO] {self.dataset_type} dataset loaded with sequence length {self.seq_len}.')

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, 'tr')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        id, text, joint = batch['id'], batch['text'], batch['joints']

        joint_len = [len(j) for j in joint]
       
        joint_input = pad_sequence(joint, batch_first = True)
        joint_input = rearrange(joint_input, 'b t v c -> b t (v c)')
        
        z = self.model.encode(joint_input)

        recon = self.model.generate(z)
        recon = rearrange(recon, 'b t (v c) -> b t v c', c = 2)
        recon = recon.cpu().detach()

        generated = []
        for r, l in zip(recon, joint_len):
            generated.append(r[:l])

        joint = [j.cpu().detach() for j in joint]
        
        return {
            'id': id,
            'text': text,
            'origin': joint,
            'generated': generated,
        }

    def validation_epoch_end(self, outputs):
        H, W = 256, 256
        S = self.S
        output = outputs[0] # select only one example
                
        # postprocessing for a full joint encoded and decoded output
        _joint = postprocess(output['_joint'], H, W, S)
        _decoded = postprocess(output['_decoded'], H, W, S)
        _text = output['_text']
        
        # save video outputs
        if self.current_epoch != 0:    
            vid_save_path = os.path.join(self.logger.save_dir, self.logger.name, f'version_{str(self.logger.version)}', 'vid_outputs', str(self.global_step))        
            if not os.path.exists(vid_save_path):
                os.makedirs(vid_save_path)
            save_sign_video(os.path.join(vid_save_path, 'full_outputs.mp4'), _decoded, _joint, _text, H, W)  
            
    def test_epoch_end(self, outputs):
        H, W = 256, 256
        S = self.S

        id_list, text_list, generated_list, origin_list = [], [], [], []
        for output in outputs:
            id = output['id']
            text = output['text']
            generated = output['generated']
            origin = output['origin']

            id_list += id
            text_list += text
            generated_list += generated
            origin_list += origin

        if self.logger.save_dir != None:
            save_path = os.path.join(self.logger.save_dir, self.logger.name, f'version_{str(self.logger.version)}/test_outputs')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # save generated joint outputs
            outputs = {
                'outputs': generated_list,
                'reference': origin_list,
                'texts': text_list,
                'ids': id_list
            }

            torch.save(outputs, os.path.join(save_path, 'outputs.pt'))
            
            _iter = zip(origin_list[:self.num_save], generated_list[:self.num_save], id_list[:self.num_save], text_list[:self.num_save])
            for j, d, id, text in _iter:               
                origin = postprocess(j, H, W, S)
                generated = postprocess(d, H, W, S)
                save_sign_video(os.path.join(save_path, f'{id}.mp4'), generated, origin, text, H, W)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr = self.lr, amsgrad = True)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optim,
            mode = 'min',
            factor = 0.5,
            patience = 10,
            cooldown = 10,
            min_lr = 1e-6,
            verbose = True
        )
        
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': sched,
                'monitor': 'val/loss',
                'frequency': self.trainer.check_val_every_n_epoch
            },
        }

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_worker, collate_fn = self._collate_fn)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_worker, collate_fn = self._collate_fn)

    def test_dataloader(self):
        return DataLoader(self.testset_full, batch_size = self.batch_size, shuffle = False, num_workers = self.num_worker, collate_fn = self._collate_fn)

    def _collate_fn(self, batch):
        id_list, text_list, joint_feat_list, frame_len_list = [], [], [], []
        sorted_batch = sorted(batch, key = lambda x: x['frame_len'], reverse = True)
        for data in sorted_batch:
            id_list.append(data['id'])
            text_list.append(data['text'])
            joint_feat_list.append(data['joint_feats'])
            frame_len_list.append(data['frame_len'])
        # joint_feats_tensor = pad_sequence(joint_feat_list, batch_first = True, padding_value = 0.)

        return {
            'id': id_list,
            'text': text_list,
            'joints': joint_feat_list,
            # 'frame_len': frame_len_list
        }

    
def main(hparams):
    # random seed
    pl.seed_everything(hparams.seed)

    # define lightning module
    if not hparams.finetuning:
        module = SpatialVAEModule(**vars(hparams))
    else:
        module = SpatialVAEModule.load_from_checkpoint(hparams.ckpt)
        
        # manually change module paramters
        module.dataset_type = hparams.dataset_type
        module.train_path = hparams.train_path
        module.valid_path = hparams.valid_path
        module.test_path = hparams.test_path
        module.batch_size = hparams.batch_size
        module.num_worker = hparams.num_workers
        module.check_val_every_n_epoch = hparams.check_val_every_n_epoch
        module.max_epochs = hparams.max_epochs

        # module.model.vq_layer.kl_weight = hparams.kl_weight
        # module.model.anneal_rate = hparams.anneal_rate
    
        
        print(f'[INFO] Pretrained model loaded from {hparams.ckpt}')
        
        # prevent resuming training
        hparams.ckpt = None
   
    # callbacks
    early_stopping_callback = EarlyStopping(
        monitor = 'val/loss', 
        patience = 100, 
        mode = 'min', 
        verbose = True
    )
    ckpt_callback = ModelCheckpoint(
        filename = 'epoch={epoch}-val_loss={val/loss:.2f}', 
        monitor = 'val/loss', 
        save_last = True, 
        save_top_k = 1, 
        mode = 'min', 
        verbose = True,
        auto_insert_metric_name = False
    )

    callback_list = [ckpt_callback]
    if hparams.use_early_stopping:
        callback_list.append(early_stopping_callback)

    # define logger
    logger = TensorBoardLogger("slp_logs", name = hparams.log_name)
    hparams.logger = logger
    
    # define trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams, 
        callbacks = callback_list
    )

    if not hparams.test:
        trainer.fit(module, ckpt_path = hparams.ckpt if hparams.ckpt != None else None)
        if not hparams.fast_dev_run:
            trainer.test(module)
    else:
        assert hparams.ckpt != None, 'Trained checkpoint must be provided.'
        trainer.test(module, ckpt_path = hparams.ckpt)


if __name__=='__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--dataset_type', default = 'how2sign')
    parser.add_argument('--train_path', default = '/home/ejhwang/projects/how2sign/how2sign_realigned_train.csv')
    parser.add_argument('--valid_path', default = '/home/ejhwang/projects/how2sign/how2sign_realigned_val.csv')
    parser.add_argument('--test_path', default = '/home/ejhwang/projects/how2sign/how2sign_realigned_test.csv')
    parser.add_argument("--fast_dev_run", action = "store_true")
    parser.add_argument('--seq_len', type = int, default = 32)
    parser.add_argument("--batch_size", type = int, default = 2)
    parser.add_argument("--num_workers", type = int, default = 0)
    parser.add_argument("--max_epochs", type = int, default = 500)
    parser.add_argument('--check_val_every_n_epoch', type = int, default = 3)
    parser.add_argument('--accelerator', default = 'cpu')
    parser.add_argument('--devices', nargs = '+', type = int, default = None)
    parser.add_argument('--strategy', default = None)
    parser.add_argument('--num_save', type = int, default = 5)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--ckpt', default = None)
    parser.add_argument('--finetuning', action = 'store_true')
    parser.add_argument('--test', action = 'store_true')
    parser.add_argument('--use_early_stopping', action = 'store_true')
    parser.add_argument('--log_name', type = str, default = 'spvae')

    parser = SpatialVAE.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)

import math
import os
import random
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from backup.data import load_data
from fid import calculate_frechet_distance
from modules.model.layers import PoseEmbLayer, PoseGenerator, PositionalEncoding
from render import save_sign_video, save_sign_video_batch
from backup.utils import noised, postprocess


def get_mean_cov(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return x.mean(axis = 0), np.cov(x, rowvar = False)


def cal_mean_for_length(x, device):
    mask = torch.tensor([len(x) for _x in x], device = device)
    x = pad_sequence(x, batch_first = True, padding_value = 0.)
    # x = x.sum(1) / mask.unsqueeze(1)
    x = rearrange(x, 'b t d -> (b t) d')
    
    return x


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_head,
        num_hidden_layers,
        intermediate_size,
        layer_dims,
        noise_rate,
        dropout,
        seq_len,
        **kwargs
    ):
        super().__init__()

        enc_layer_dims = layer_dims + [hidden_size]
        dec_layer_dims = [hidden_size] + list(reversed(layer_dims))

        pre = PoseEmbLayer(enc_layer_dims, 'relu', dropout)
        generator = PoseGenerator(dec_layer_dims, 'relu', dropout)

        pos_emb = PositionalEncoding(hidden_size, max_len = seq_len + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = hidden_size,
            nhead = num_attention_head,
            dim_feedforward = intermediate_size,
            dropout = dropout,
            batch_first = True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = hidden_size,
            nhead = num_attention_head,
            dim_feedforward = intermediate_size,
            dropout = dropout,
            batch_first = True
        )
        
        encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        decoder = nn.TransformerDecoder(decoder_layer, num_hidden_layers)

        self.pre = pre
        self.generator = generator
        self.encoder = encoder
        self.decoder = decoder
        self.pos_emb = pos_emb

        self.noise_rate = noise_rate
        self.hidden_size = hidden_size
        self.seq_len = seq_len

    def forward(self, x, device = 'cpu'):
        b, n_frame, _ = x.size()

        if self.training and self.noise_rate > 0.0:
            noised_x = noised(x, self.noise_rate)
        else:
            noised_x = x

        pre_x = self.pre(noised_x)
        z = self.encoder(pre_x)

        # temporal pooling
        z = z.mean(1)

        time_queries = torch.zeros(n_frame, b, self.hidden_size, device = device)
        time_queries = self.pos_emb(time_queries)
        time_queries = rearrange(time_queries, 't b d -> b t d')
        
        memory = z.unsqueeze(1)
        logits = self.decoder(time_queries, memory)
        
        recon_x = self.generator(logits)

        loss = F.binary_cross_entropy(recon_x, x)

        return loss, recon_x

    def generate(self, z_list, device = 'cpu'):
        generated_list = []
        for z in z_list:
            b, n_frame = z.size(0), self.seq_len

            time_queries = torch.zeros(n_frame, b, self.hidden_size, device = device)
            time_queries = self.pos_emb(time_queries)
            time_queries = rearrange(time_queries, 't b d -> b t d')
            
            memory = z.unsqueeze(1)
            logits = self.decoder(time_queries, memory)
            generated = self.generator(logits)
            generated = rearrange(generated, 'b t d -> (b t) d')

            generated_list.append(generated)
        
        return generated_list

    @torch.no_grad()
    def encode(
        self, 
        input_joints,
        device = 'cpu'
    ):
        assert type(input_joints) == list, 'Inputs must be list of tensors.'

        z_list = []
        for joint_feat in input_joints:
            _rest = joint_feat.size(0) % self.seq_len
            if _rest > 0:
                _pruned_joint_feat = joint_feat[:-_rest, :]
            else:
                _pruned_joint_feat = joint_feat
            
            # chunking
            _chunk = _pruned_joint_feat.size(0) // self.seq_len
            
            _chunk_feat = torch.chunk(_pruned_joint_feat, _chunk, dim = 0)
            _chunk_feat = torch.stack(_chunk_feat)
            _chunk_feat = _chunk_feat.to(device)

            _chunk_feat = self.pre(_chunk_feat)
            z = self.encoder(_chunk_feat)
            
            z = z.mean(1)
            
            z_list.append(z)
        
        return z_list

    def eval_fgd(self, generated, reference, device = 'cpu'):
        assert type(generated) == list, 'Inputs must be list of tensors.'

        pred_z = self.encode(generated, device = device)
        real_z = self.encode(reference, device = device)

        pred_z, real_z = map(lambda x: cal_mean_for_length(x, device), [pred_z, real_z])
        # pred_z, real_z = map(lambda x: torch.vstack(x), [pred_z, real_z])
                
        (pred_mu, pred_sigma), (real_mu, real_sigma) = \
            map(lambda x: get_mean_cov(x), [pred_z, real_z])
        
        score = calculate_frechet_distance(pred_mu, pred_sigma, real_mu, real_sigma)
        
        return score

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('sp_vqvae')
        parser.add_argument('--layer_dims', nargs = '+', type = int, default = [240, 1024, 512])
        parser.add_argument('--noise_rate', type = float, default = 0.001)
        parser.add_argument('--hidden_size', type = int, default = 768)
        parser.add_argument('--num_attention_head', type = int, default = 12)
        parser.add_argument('--num_hidden_layers', type = int, default = 8)
        parser.add_argument('--intermediate_size', type = int, default = 2048)
        parser.add_argument('--dropout', type = float, default = 0.1)
        
        return parent_parser


class TransformerAutoencoderModule(pl.LightningModule):
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
        self.model = TransformerAutoencoder(seq_len = seq_len, **kwargs)

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

    def setup(self, stage):        
        if stage == 'fit':
            self.trainset, self.validset, _ = \
                load_data(
                    self.dataset_type, 
                    self.train_path, 
                    self.valid_path, 
                    self.test_path, 
                    seq_len = self.seq_len, 
                    min_seq_len = self.seq_len
                )
            _, self.validset_full, _ = \
                load_data(
                    self.dataset_type, 
                    self.train_path, 
                    self.valid_path, 
                    self.test_path, 
                    min_seq_len = self.seq_len
                )
        else:
            _, self.testset, _ = \
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

    def _common_step(self, batch, stage):
        id = batch['id']
        text = batch['text']
        joints = batch['joint_feats']

        joints = pad_sequence(joints, batch_first = True, padding_value = 0.)

        joints = rearrange(joints, 'b t v c -> b t (v c)')

        loss, recon_x = self.model(joints, device = self.device)
        
        self.log(f'{stage}/loss', loss, batch_size = self.batch_size)

        recon_x = rearrange(recon_x, 'b t (v c) -> (b t) v c', c = 2)
        joints = rearrange(joints, 'b t (v c) -> (b t) v c', c = 2)

        if stage == 'tr':
            return loss
        
        # full encode and decode
 
        full_dataset = self.validset_full
        
        _vf_data = random.choice(full_dataset)
        
        _joint = _vf_data['joint_feats']
        _text = _vf_data['text']
        _id = _vf_data['id']
        
        _joint = rearrange(_joint, 't v c -> t (v c)')
        
        _z = self.model.encode([_joint], self.device)
        
        _decoded = self.model.generate(_z, device = self.device)
        _decoded = _decoded[0]
        
        _joint = rearrange(_joint, 't (v c) -> t v c', c = 2)
        _decoded = rearrange(_decoded, 't (v c) -> t v c', c = 2)
        
        return {
            'loss': loss,
            'text': text,
            'id': id,
            'origin': joints.cpu().detach(),
            'recon_x': recon_x.cpu().detach(),
            '_joint': _joint.cpu().detach(),
            '_decoded': _decoded.cpu().detach(),
            '_text': _text,
            '_id': _id
        }

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, 'tr')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        id = batch['id']
        text = batch['text']
        joints = batch['joint_feats']
        joints = [rearrange(j, 't v c -> t (v c)') for j in joints]
        
        encoded = self.model.encode(joints, device = self.device)
        generated = self.model.generate(encoded, device = self.device)

        # self.model.eval_fgd(generated, joints, device = self.device)

        joints = [j.cpu() for j in joints]
        generated = [g.cpu() for g in generated]
        
        return {
            'id': id,
            'text': text,
            'joints': joints,
            'decoded': generated
        }

    def validation_epoch_end(self, outputs):
        H, W = 256, 256
        S = self.S
        output = outputs[0] # select only one example
        
        recon, origin = map(lambda t: postprocess(t, H, W, S), [output['recon_x'], output['origin']])
        recon, origin = map(lambda t: rearrange(t, '(b t) d -> b t d', b = self.batch_size), [recon, origin])
        
        recon = recon[:self.num_save, :, :]
        origin = origin[:self.num_save, :, :]
        
        id = output['id'][:self.num_save]
        text = output['text'][:self.num_save]
 
        # postprocessing for a full joint encoded and decoded output
        _joint = postprocess(output['_joint'], H, W, S)
        _decoded = postprocess(output['_decoded'], H, W, S)
        _text = output['_text']
        
        # save video outputs
        if self.current_epoch != 0:    
            vid_save_path = os.path.join(self.logger.save_dir, self.logger.name, f'version_{str(self.logger.version)}', 'vid_outputs', str(self.global_step))        
            
            save_sign_video_batch(vid_save_path, recon, origin, text, id, H, W)
            
            if not os.path.exists(vid_save_path):
                os.makedirs(vid_save_path)
            
            save_sign_video(os.path.join(vid_save_path, 'full_outputs.mp4'), _decoded, _joint, _text, H, W)  
            
    def test_epoch_end(self, outputs):
        H, W = 256, 256
        S = self.S

        id_list, text_list, decoded_list, joints_list = [], [], [], []
        for output in outputs:
            id = output['id']
            text = output['text']
            decoded = output['decoded']
            joints = output['joints']

            id_list += id
            text_list += text
            decoded_list += decoded
            joints_list += joints

        if self.logger.save_dir != None:
            save_path = os.path.join(self.logger.save_dir, self.logger.name, f'version_{str(self.logger.version)}/test_outputs')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # save generated joint outputs
            outputs = {
                'outputs': decoded_list,
                'texts': text_list,
                'ids': id_list
            }
            torch.save(outputs, os.path.join(save_path, 'outputs.pt'))
            
            _iter = zip(joints_list[:self.num_save], decoded_list[:self.num_save], id_list[:self.num_save], text_list[:self.num_save])
            for j, d, id, text in _iter:
                j, d = map(lambda x: rearrange(x, 't (v c) -> t v c', c = 2), [j, d])
                origin = postprocess(j, H, W, S)
                generated = postprocess(d, H, W, S)
                save_sign_video(os.path.join(save_path, f'{id}.mp4'), generated, origin, text, H, W)
            
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr = self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optim,
            mode = 'min',
            factor = 0.5,
            patience = 10,
            cooldown = 10,
            min_lr = 1e-9,
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
        return DataLoader(self.testset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_worker, collate_fn = self._collate_fn)

    def _collate_fn(self, batch):
        id_list, text_list, joint_feat_list, frame_len_list = [], [], [], []
        sorted_batch = sorted(batch, key = lambda x: x['frame_len'], reverse = True)
        for data in sorted_batch:
            id_list.append(data['id'])
            text_list.append(data['text'])
            joint_feat_list.append(data['joint_feats'])
            frame_len_list.append(data['frame_len'])

        return {
            'id': id_list,
            'text': text_list,
            'joint_feats': joint_feat_list,
            'frame_len': frame_len_list
        }

    def get_callback_fn(self, monitor = 'val/loss', patience = 50):
        early_stopping_callback = EarlyStopping(
            monitor = monitor, 
            patience = patience, 
            mode = 'min', 
            verbose = True
        )
        ckpt_callback = ModelCheckpoint(
            filename = 'epoch={epoch}-val_loss={val/loss:.2f}', 
            monitor = monitor, 
            save_last = True, 
            save_top_k = 1, 
            mode = 'min', 
            verbose = True,
            auto_insert_metric_name = False
        )
        return early_stopping_callback, ckpt_callback

    def get_logger(self, type = 'tensorboard', name = 'slp'):
        if type == 'tensorboard':
            logger = TensorBoardLogger("slp_logs", name = name)
        else:
            raise NotImplementedError
        return logger


def main(hparams):
    # random seed
    pl.seed_everything(hparams.seed)

    # define lightning module
    if not hparams.finetuning:
        module = TransformerAutoencoderModule(**vars(hparams))
    else:
        module = TransformerAutoencoderModule.load_from_checkpoint(hparams.ckpt)
        
        # manually change module paramters
        module.dataset_type = hparams.dataset_type
        module.train_path = hparams.train_path
        module.valid_path = hparams.valid_path
        module.test_path = hparams.test_path
        module.batch_size = hparams.batch_size
        module.num_worker = hparams.num_workers
        module.check_val_every_n_epoch = hparams.check_val_every_n_epoch
        module.max_epochs = hparams.max_epochs
        
        print(f'[INFO] Pretrained model loaded from {hparams.ckpt}')
        
        # prevent resuming training
        hparams.ckpt = None
   
    early_stopping, ckpt = module.get_callback_fn('val/loss', 30)
    
    callbacks_list = [ckpt]

    if hparams.use_early_stopping:
        callbacks_list.append(early_stopping)
    
    logger = module.get_logger('tensorboard', name = 'tfae')
    hparams.logger = logger
    
    # define trainer
    trainer = pl.Trainer.from_argparse_args(hparams, callbacks = callbacks_list)

    if not hparams.test:
        trainer.fit(module, ckpt_path = hparams.ckpt if hparams.ckpt != None else None)
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
    parser.add_argument('--check_val_every_n_epoch', type = int, default = 5)
    parser.add_argument('--accelerator', default = 'cpu')
    parser.add_argument('--devices', nargs = '+', type = int, default = None)
    parser.add_argument('--strategy', default = None)
    parser.add_argument('--num_save', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--ckpt', default = None)
    parser.add_argument('--finetuning', action = 'store_true')
    parser.add_argument('--test', action = 'store_true')
    parser.add_argument('--use_early_stopping', action = 'store_true')

    parser = TransformerAutoencoder.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams) 

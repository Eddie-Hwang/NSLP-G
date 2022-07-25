import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from layers import PositionalEncoding
from render import save_sign_video, save_sign_video_batch
from slp_trainer import SignLanguageProductionTrainer
from tokenizer import (HugTokenizer, SimpleTokenizer, build_vocab_from_phoenix,
                       white_space_tokenizer)
from train_spavae import SpatialVAE, SpatialVAEModule
from utils import noised, postprocess


class NonAutoRegressiveSLP(nn.Module):
    def __init__(
        self,
        vae,
        hidden_size,
        num_attention_head,
        num_hidden_layers,
        intermediate_size,
        dropout,
        text_vocab_size,
        max_seq_len,
        **kwargs
    ):
        super().__init__()

        latent_dim = vae.latent_dim

        text_emb = nn.Embedding(text_vocab_size, hidden_size)
        
        pos_emb = PositionalEncoding(hidden_size, max_len = max_seq_len)

        model = nn.Transformer(
            d_model = hidden_size,
            nhead = num_attention_head,
            num_encoder_layers = num_hidden_layers,
            dim_feedforward = intermediate_size,
            dropout = dropout,
            batch_first = True
        )

        to_latent = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, latent_dim)
        )

        self.model = model
        self.to_latent = to_latent
        self.text_emb = text_emb
        self.pos_emb = pos_emb
        self.vae = vae
        self.hidden_size = hidden_size

    def forward(
        self,
        text_input_ids,
        text_pad_mask,
        joint_inputs,
        joint_pad_mask,
        return_outs = False,
        device = 'cpu'
    ):
        embed_text = self.text_emb(text_input_ids)
        embed_text = self.pos_emb(rearrange(embed_text, 'b t d -> t b d'))
        embed_text = rearrange(embed_text, 't b d -> b t d')

        b, n_frame, _ = joint_inputs.size()
        
        time_qeuries = torch.zeros(n_frame, b, self.hidden_size, device = device)
        time_qeuries = self.pos_emb(time_qeuries)
        time_qeuries = rearrange(time_qeuries, 't b d -> b t d')
        
        logits = self.model(
            src = embed_text,
            tgt = time_qeuries,
            src_key_padding_mask = text_pad_mask,
            tgt_key_padding_mask = joint_pad_mask
        )

        z = self.to_latent(logits)

        outs = self.vae.generate(z)
        
        if return_outs:
            return outs

        loss = F.mse_loss(outs, joint_inputs, reduction = 'none')
        loss = loss.sum(-1)
        loss.masked_fill_(joint_pad_mask, 0.)
        loss = loss.mean()
        
        return loss

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('nslpg')   
        parser.add_argument('--hidden_size', type = int, default = 512)
        parser.add_argument('--num_attention_head', type = int, default = 8)
        parser.add_argument('--num_hidden_layers', type = int, default = 2)
        parser.add_argument('--intermediate_size', type = int, default = 1024)
        parser.add_argument('--max_seq_len', type = int, default = 512)
        parser.add_argument('--dropout', type = float, default = 0.)
        parser.add_argument('--vae_ckpt', type = str, default = './slp_logs/spavae/phoenix/checkpoints/last.ckpt')
        return parent_parser  


class NonAutoRegressiveSLPModule(SignLanguageProductionTrainer):
    def __init__(
        self,
        train_path,
        vae_ckpt,
        **kwargs
    ):
        text_vocab = build_vocab_from_phoenix(train_path, white_space_tokenizer, mode = 'text')
        text_tokenizer = SimpleTokenizer(white_space_tokenizer, text_vocab)
        
        text_vocab_size = text_tokenizer.vocab_size
        text_pad_token = text_tokenizer.pad_token

        vae = SpatialVAEModule.load_from_checkpoint(vae_ckpt)
        vae.eval()
        vae.freeze()
        vae = vae.model

        model = NonAutoRegressiveSLP(
            vae = vae, 
            text_vocab_size = text_vocab_size,
            **kwargs
        )
        
        super().__init__(
            model = model, 
            tokenizer = text_tokenizer, 
            train_path = train_path,
            min_seq_len = -1,
            **kwargs
        )

    def _common_step(self, batch, stage):
        id, text, joint = batch['id'], batch['text'], batch['joints']

        text_input_ids, text_pad_mask = self.tokenizer.encode(
            text, 
            padding = True,
            add_special_tokens = True,
            device = self.device
        )
        text_pad_mask = ~(text_pad_mask.bool()).to(self.device)

        joint_pad_mask = [torch.ones(j.size(0)) for j in joint]
        joint_pad_mask = pad_sequence(joint_pad_mask, batch_first = True)
        joint_pad_mask = ~(joint_pad_mask.bool()).to(self.device)
        
        joint_input = pad_sequence(joint, batch_first = True)
        joint_input = rearrange(joint_input, 'b t v c -> b t (v c)')

        loss = self.model(
            text_input_ids = text_input_ids,
            text_pad_mask = text_pad_mask,
            joint_inputs = joint_input,
            joint_pad_mask = joint_pad_mask,
            device = self.device
        )

        self.log(f'{stage}/loss', loss, batch_size = self.batch_size)

        if stage == 'tr':
            return loss

        idx = torch.randperm(text_input_ids.size(0))[:self.num_save]
        
        generated = self.model(
            text_input_ids = text_input_ids[idx],
            text_pad_mask = text_pad_mask[idx],
            joint_inputs = joint_input[idx],
            joint_pad_mask = joint_pad_mask[idx],
            return_outs = True,
            device = self.device
        )
        generated = rearrange(generated, 'b t (v c) -> b t v c', c = 2)
        generated = generated.cpu()

        origin = list(joint[i].cpu() for i in idx)
        text = list(text[i] for i in idx)
        name = list(id[i] for i in idx)
        
        return {
            'loss': loss,
            'name': name,
            'text': text,
            'origin': origin,
            'generated': generated,
        }

    def _common_epoch_end(self, outputs, stage):
        H, W = 256, 256
        S = self.S

        output = outputs[0] # select only one example
    
        origin = output['origin']
        generated = output['generated']
        text = output['text']
        name = output['name']

        processed_origin, processed_generated = [], []
        for ori, gen in zip(origin, generated):
            processed_ori, processed_gen = map(lambda t: postprocess(t, H, W, S), [ori, gen])
            
            processed_origin.append(processed_ori)
            processed_generated.append(processed_gen)
    
        if self.current_epoch != 0:
            vid_save_path = os.path.join(self.logger.save_dir, self.logger.name, f'version_{str(self.logger.version)}', 'vid_outputs', str(self.global_step))
            
            if not os.path.exists(vid_save_path):
                os.makedirs(vid_save_path)
            
            for n, t, g, o in zip(name, text, processed_generated, processed_origin):
                save_sign_video(fpath = os.path.join(vid_save_path, f'{n}.mp4'), hyp = g, ref = o, sent = t, H = H, W = W)
        
        
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


def main(hparams):
    pl.seed_everything(hparams.seed)
    
    module = NonAutoRegressiveSLPModule(**vars(hparams))
    
    early_stopping, ckpt = module.get_callback_fn('val/loss', 50)
    
    callbacks_list = [ckpt]

    if hparams.use_early_stopping:
        callbacks_list.append(early_stopping)
    
    logger = module.get_logger('tensorboard', name = 'narslp')
    hparams.logger = logger
    
    trainer = pl.Trainer.from_argparse_args(hparams, callbacks = callbacks_list)
    trainer.fit(module)


if __name__=='__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument("--fast_dev_run", action = "store_true")
    parser.add_argument('--dataset_type', default = 'phoenix')
    parser.add_argument('--train_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.train')
    parser.add_argument('--valid_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.dev')
    parser.add_argument('--test_path', default = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.test')
    parser.add_argument("--batch_size", type = int, default = 2)
    parser.add_argument("--num_workers", type = int, default = 0)
    parser.add_argument("--max_epochs", type = int, default = 500)
    parser.add_argument('--check_val_every_n_epoch', type = int, default = 3)
    parser.add_argument('--accelerator', default = 'gpu')
    parser.add_argument('--devices', nargs = '+', type = int, default = [1])
    parser.add_argument('--strategy', default = None)
    parser.add_argument('--num_save', type = int, default = 3)
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--use_early_stopping', action = 'store_true')
    parser.add_argument('--gradient_clip_val', type = float, default = 0.0)

    parser = NonAutoRegressiveSLP.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)


'''
(train on phoenix)
python scripts/train_nslpg.py \
    --accelerator gpu --devices 0 \
    --num_worker 8 --batch_size 64
'''

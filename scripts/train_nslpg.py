import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data import load_data
from layers import PositionalEncoding
from render import save_sign_video
from tokenizer import (HugTokenizer, SimpleTokenizer, build_vocab_from_phoenix,
                       white_space_tokenizer)
from train_spavae import SpatialVAEModule
from utils import postprocess


def set_whitespace_tokenizer(train_path, mode = 'text', **kwargs):
    _vocab = build_vocab_from_phoenix(train_path, white_space_tokenizer, mode = mode)
    _tokenizer = SimpleTokenizer(white_space_tokenizer, _vocab)

    return {
        'tokenizer': _tokenizer,
        'pad_token': _tokenizer.pad_token,
        'start_token': _tokenizer.start_token,
        'end_token': _tokenizer.end_token,
        'vocab': _vocab,
        'vocab_size': len(_vocab)
    }


def set_hug_tokenizer(tokenizer_fpath, **kwargs):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    _tokenizer = HugTokenizer(tokenizer_fpath)

    return {
        'tokenizer': _tokenizer, 
        'pad_token': _tokenizer.pad_token,
        'start_token': _tokenizer.start_token,
        'end_token': _tokenizer.end_token,
        'vocab_size': _tokenizer.vocab_size
    }


def set_tokenizer_dict():
    return {
        'whitespace': set_whitespace_tokenizer,
        'wordpiece': set_hug_tokenizer,
        'bpe': set_hug_tokenizer,
    }


def load_pretrained_vae(finetuning, ckpt):
    vae = SpatialVAEModule.load_from_checkpoint(ckpt)
    if not finetuning:
        vae.eval()
        vae.freeze()
    vae = vae.model

    return vae


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
        gloss_supervision,
        gloss_vocab_size,
        alpha,
        **kwargs
    ):
        super().__init__()

        latent_dim = vae.latent_dim

        text_emb = nn.Embedding(text_vocab_size, hidden_size)
        
        pos_emb = PositionalEncoding(hidden_size, max_len = max_seq_len)

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

        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_hidden_layers)

        if gloss_supervision:
            gloss_decoder_layer = nn.TransformerDecoderLayer(
                d_model = hidden_size,
                nhead = num_attention_head,
                dim_feedforward = intermediate_size,
                dropout = dropout,
                batch_first = True
            )
            
            self.gloss_decoder = nn.TransformerDecoder(gloss_decoder_layer, num_hidden_layers)
            self.to_gloss = nn.Linear(hidden_size, gloss_vocab_size)
            self.gloss_emb = nn.Embedding(gloss_vocab_size, hidden_size)

        self.to_latent = nn.Linear(hidden_size, latent_dim)

        # self.model = model
        self.text_emb = text_emb
        self.pos_emb = pos_emb
        self.vae = vae
        self.hidden_size = hidden_size
        self.gloss_supervision = gloss_supervision
        self.alpha = alpha

    def forward(
        self,
        text_input_ids,
        text_pad_mask,
        joint_inputs,
        joint_pad_mask,
        return_outs = False,
        gloss_input_ids = None,
        gloss_pad_mask = None,
        gloss_pad_token = None,
        device = 'cpu'
    ):
        embed_text = self.text_emb(text_input_ids)
        embed_text = self.pos_emb(rearrange(embed_text, 'b t d -> t b d'))
        embed_text = rearrange(embed_text, 't b d -> b t d')

        b, n_frame, _ = joint_inputs.size()
        
        time_qeuries = torch.zeros(n_frame, b, self.hidden_size, device = device)
        time_qeuries = self.pos_emb(time_qeuries)
        time_qeuries = rearrange(time_qeuries, 't b d -> b t d')
        
        enc_outs = self.encoder(embed_text, src_key_padding_mask = text_pad_mask)

        logits = self.decoder(time_qeuries, enc_outs, tgt_key_padding_mask = joint_pad_mask)

        z = self.to_latent(logits)

        outs = self.vae.generate(z)
        
        if return_outs:
            return outs
        
        if self.gloss_supervision:
            gloss_embed = self.gloss_emb(gloss_input_ids)
            
            gloss_logits = self.gloss_decoder(gloss_embed, enc_outs, tgt_key_padding_mask = gloss_pad_mask)
            gloss_logits = self.to_gloss(gloss_logits)
            gloss_logits = rearrange(gloss_logits, 'b t d -> b d t')
            
            gloss_loss = F.cross_entropy(gloss_logits, gloss_input_ids, ignore_index = gloss_pad_token)
        else:
            gloss_loss = 0.

        mse_loss = F.mse_loss(outs, joint_inputs, reduction = 'none')
        mse_loss = mse_loss.sum(-1)
        mse_loss.masked_fill_(joint_pad_mask, 0.)
        mse_loss = mse_loss.mean()

        alpha = self.alpha
        
        loss = mse_loss + alpha * gloss_loss

        return loss, mse_loss, gloss_loss

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('nslpg')   
        parser.add_argument('--hidden_size', type = int, default = 512)
        parser.add_argument('--num_attention_head', type = int, default = 8)
        parser.add_argument('--num_hidden_layers', type = int, default = 4)
        parser.add_argument('--intermediate_size', type = int, default = 1024)
        parser.add_argument('--max_seq_len', type = int, default = 512)
        parser.add_argument('--min_seq_len', type = int, default = 32)
        parser.add_argument('--dropout', type = float, default = 0.)
        parser.add_argument('--vae_ckpt', type = str, default = './slp_logs/spavae/phoenix/checkpoints/last.ckpt')
        parser.add_argument('--tokenizer_type', type = str, default = 'whitespace')
        parser.add_argument('--tokenizer_fpath', type = str, default = None)
        parser.add_argument('--vae_ft', action = 'store_true')
        parser.add_argument('--gloss_supervision', action = 'store_true')
        parser.add_argument('--alpha', type = float, default = 0.)
        return parent_parser  


class NonAutoRegressiveSLPModule(pl.LightningModule):
    def __init__(
        self,
        dataset_type,
        num_save,
        train_path,
        valid_path,
        test_path,
        batch_size,
        num_workers,
        lr,
        vae_ckpt,
        tokenizer_type,
        tokenizer_fpath,
        min_seq_len,
        max_seq_len,
        vae_ft,
        save_vids,
        gloss_supervision,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        tokenizer_dict = set_tokenizer_dict()
        
        _tokenizer = tokenizer_dict[tokenizer_type](
            train_path = train_path, 
            tokenizer_fpath = tokenizer_fpath, 
        )
        
        text_vocab_size = _tokenizer['vocab_size']
        text_pad_token = _tokenizer['pad_token']
        text_tokenizer = _tokenizer['tokenizer']

        if dataset_type == 'how2sign':
            assert gloss_supervision != True, 'How2Sign does not yet contain gloss.'

        if gloss_supervision:
            _gloss_tokenizer = set_whitespace_tokenizer(train_path, 'gloss')
            gloss_vocab_size = _gloss_tokenizer['vocab_size']
            self.gloss_pad_token = _gloss_tokenizer['pad_token']
            self.gloss_tokenizer = _gloss_tokenizer['tokenizer']

        vae = load_pretrained_vae(vae_ft, ckpt = vae_ckpt)
        model = NonAutoRegressiveSLP(
            vae = vae, 
            text_vocab_size = text_vocab_size, 
            gloss_vocab_size = gloss_vocab_size if gloss_supervision else None,
            max_seq_len = max_seq_len,
            gloss_supervision = gloss_supervision,
            **kwargs
        )

        self.model = model
        self.tokenizer = text_tokenizer
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        # dataset related parameters
        self.dataset_type = dataset_type
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        # Training related paramters
        self.num_save = num_save
        self.batch_size = batch_size
        self.num_worker = num_workers
        self.lr = lr
        self.save_vids = save_vids
        self.gloss_supervision = gloss_supervision

    def setup(self, stage):
        # load dataset
        self.trainset, self.validset, self.testset = \
            load_data(
                self.dataset_type, 
                self.train_path, 
                self.valid_path, 
                self.test_path, 
                seq_len = self.max_seq_len, 
                min_seq_len = self.min_seq_len
            )
        
        print(f'[INFO] {self.dataset_type} dataset loaded with sequence length {self.max_seq_len}.')

        if self.dataset_type == 'how2sign':
            self.S = 3
        else:
            self.S = 1.5        

    def _common_step(self, batch, stage):
        id, text, gloss, joint = \
            batch['id'], batch['text'], batch['gloss'], batch['joints']

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

        if self.gloss_supervision:
            gloss_input_ids, gloss_pad_mask = self.gloss_tokenizer.encode(
                gloss,
                padding = True,
                add_special_tokens = False,
                device = self.device
            )
            gloss_pad_mask = ~(gloss_pad_mask.bool()).to(self.device)
        else:
            gloss_input_ids, gloss_pad_mask = None, None
        
        loss, mse_loss, gloss_loss = self.model(
            text_input_ids = text_input_ids,
            text_pad_mask = text_pad_mask,
            joint_inputs = joint_input,
            joint_pad_mask = joint_pad_mask,
            gloss_input_ids = gloss_input_ids,
            gloss_pad_mask = gloss_pad_mask,
            gloss_pad_token = self.gloss_pad_token if self.gloss_supervision else None,
            device = self.device
        )

        self.log(f'{stage}/loss', loss, batch_size = self.batch_size)
        self.log(f'{stage}/mse_loss', mse_loss, batch_size = self.batch_size)
        self.log(f'{stage}/gloss_loss', gloss_loss, batch_size = self.batch_size)

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

        origin_list = list(joint[i].cpu() for i in idx)
        text_list = list(text[i] for i in idx)
        id_list = list(id[i] for i in idx)
        
        return {
            'loss': loss,
            'id': id_list,
            'text': text_list,
            'origin': origin_list,
            'generated': generated,
        }
            
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, 'tr')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, 'tst')

    def validation_epoch_end(self, outputs):
        H, W = 256, 256
        S = self.S

        output = outputs[0] # select only one example
    
        origin = output['origin']
        generated = output['generated']
        text = output['text']
        id = output['id']

        processed_origin, processed_generated = [], []
        for ori, gen in zip(origin, generated):
            processed_ori, processed_gen = map(lambda t: postprocess(t, H, W, S), [ori, gen])
            
            processed_origin.append(processed_ori)
            processed_generated.append(processed_gen)
    
        if self.current_epoch != 0:
            vid_save_path = os.path.join(self.logger.save_dir, self.logger.name, f'version_{str(self.logger.version)}', 'vid_outputs', str(self.global_step))
            
            if not os.path.exists(vid_save_path):
                os.makedirs(vid_save_path)
            
            for n, t, g, o in zip(id, text, processed_generated, processed_origin):
                save_sign_video(fpath = os.path.join(vid_save_path, f'{n}.mp4'), hyp = g, ref = o, sent = t, H = H, W = W)

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
            
            if not self.save_vids:
                return
            
            _iter = zip(origin_list[:self.num_save], generated_list[:self.num_save], id_list[:self.num_save], text_list[:self.num_save])
            for j, d, id, text in _iter:
                if len(j.size()) == 2:
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
            min_lr = 1e-6,
            verbose = True
        )
        
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': sched,
                'monitor': 'tr/loss',
                'frequency': self.trainer.check_val_every_n_epoch
            },
        }

    def train_dataloader(self):
        return DataLoader(
            self.trainset, 
            batch_size = self.batch_size, 
            shuffle = True, 
            num_workers = self.num_worker, 
            collate_fn = self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset, 
            batch_size = self.batch_size, 
            shuffle = False, 
            num_workers = self.num_worker, 
            collate_fn = self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset, 
            batch_size = self.batch_size, 
            shuffle = False, 
            num_workers = self.num_worker, 
            collate_fn = self._collate_fn
        )

    def _collate_fn(self, batch):
        id_list, text_list, gloss_list, joint_list = [], [], [], []
        
        sorted_batch = sorted(batch, key = lambda x: x['frame_len'], reverse = True)
        for data in sorted_batch:
            id_list.append(data['id'])
            text_list.append(data['text'])
            joint_list.append(data['joint_feats'])
            if 'gloss' in data.keys():
                gloss_list.append(data['gloss'])

        return {
            'id': id_list,
            'text': text_list,
            'joints': joint_list,
            'gloss': gloss_list
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
    pl.seed_everything(hparams.seed)
    
    module = NonAutoRegressiveSLPModule(**vars(hparams))
    
    early_stopping, ckpt = module.get_callback_fn('tr/loss', 20)
    
    callbacks_list = [ckpt]

    if hparams.use_early_stopping:
        callbacks_list.append(early_stopping)
    
    logger = module.get_logger('tensorboard', name = hparams.log_name)
    hparams.logger = logger
    
    trainer = pl.Trainer.from_argparse_args(hparams, callbacks = callbacks_list)
    if not hparams.test:
        trainer.fit(module, ckpt_path = hparams.ckpt if hparams.ckpt != None else None)
    else:
        assert hparams.ckpt != None, 'Trained checkpoint must be provided.'
        trainer.test(module, ckpt_path = hparams.ckpt)
        

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
    parser.add_argument('--log_name', type = str, default = 'narslp')
    parser.add_argument('--ckpt', default = None)
    parser.add_argument('--test', action = 'store_true')
    parser.add_argument('--save_vids', action = 'store_true')

    parser = NonAutoRegressiveSLP.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)

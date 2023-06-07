import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from modules.helpers import (create_mask, get_vocab, instantiate_from_config,
                     load_vocab, save_vocab, strings_to_indices)
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from modules.positional import PositionalEncoding
from modules.normalize import scaling_keypoints, center_keypoints
from modules.constant import return_scale_resoution
from functools import partial


def vocab(file_path, cache, min_freq=1):
    if os.path.exists(cache):
        vocab = load_vocab(file_path=cache)
    else:
        vocab = get_vocab(file_path, min_freq)
        save_vocab(vocab, file_path=cache)

    return vocab


class GaussianSeeker(pl.LightningModule):
    def __init__(
        self,
        monitor,
        first_stage_config,
        latent_model_config,
        text_file_path, 
        gloss_file_path,
        text_vocab_cache,
        gloss_vocab_cache,
        scheduler_config,
        pose_dim=512,
        first_stage_trainable=False,
        dim_feedforward=1024, 
        d_model=512, 
        nhead=4, 
        dropout=0.1, 
        activation="relu", 
        n_layers=3,
        vocab_freq=1,
        emb_dim=512,
        base_learning_rate=0.001, 
        label_smoothing=0.1,
        gloss_loss_weight=0.1,
        **kwargs
    ):
        super().__init__()

        self.learning_rate = base_learning_rate
        self.monitor = monitor
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.label_smoothing = label_smoothing
        self.gloss_loss_weight = gloss_loss_weight
        self.first_stage_trainable = first_stage_trainable
        self.pose_dim = pose_dim
        self.scheduler_config = scheduler_config

        self.text_vocab = vocab(text_file_path, text_vocab_cache, vocab_freq)
        self.gloss_vocab = vocab(gloss_file_path, gloss_vocab_cache, vocab_freq)

        self.text_emb = nn.Embedding(num_embeddings=len(self.text_vocab), embedding_dim=emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers)

        self.gloss_emb = nn.Embedding(num_embeddings=len(self.gloss_vocab), embedding_dim=emb_dim)
        gloss_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.gloss_decoder = nn.TransformerDecoder(decoder_layer=gloss_decoder_layer, num_layers=n_layers)
        self.gloss_outs = nn.Linear(d_model, len(self.gloss_vocab))

        self.pos_encoding = PositionalEncoding(d_model)

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_latent_model_stage(latent_model_config)

    def instantiate_first_stage(self, config):
        if not self.first_stage_trainable:
            model = instantiate_from_config(config)
            self.first_stage_model = model.eval()
            for param in self.first_stage_model.parameters():
                param.requires_grad = False
        else:
            self.first_stage_model = instantiate_from_config(config)

    def instantiate_latent_model_stage(self, config):
        config["params"]["input_size"] = self.d_model
        config["params"]["output_size"] = self.pose_dim
        print("Auto setting for mlp.")
        model = instantiate_from_config(config)
        self.latent_model = model

    def get_first_stage_encoding(self, inputs):
        z, _ = self.first_stage_model.encode(inputs)
        return z
    
    def decode_first_stage(self, z, mask):
        return self.first_stage_model.generate(z, mask)

    def encode(self, src, mask):
        embed = self.text_emb(src)
        embed = self.pos_encoding(embed)
        outs = self.encoder(embed, src_key_padding_mask=~mask)
        return outs
    
    def decode(self, enc_outs, trg, mask):
        embed = self.pos_encoding(trg)
        outs = self.decoder(embed, enc_outs, tgt_key_padding_mask=~mask)
        return outs
    
    def gloss_decode(self, outs, trg, mask):
        embed = self.gloss_emb(trg)
        embed = self.pos_encoding(embed)
        outs = self.gloss_decoder(embed, outs, tgt_key_padding_mask=~mask)
        outs = self.gloss_outs(outs)
        return outs

    def forward(self, text, text_mask, keypoints, keypoints_mask, gloss, gloss_mask):
        batch_size, nframes, _ = keypoints.shape
        
        enc_outs = self.encode(text, text_mask)
        time_qeuries = torch.zeros(batch_size, nframes, self.d_model, device=self.device)
        dec_outs = self.decode(enc_outs, time_qeuries, keypoints_mask)

        z_text = self.latent_model(dec_outs)
        z_pose = self.get_first_stage_encoding(keypoints)

        generated = self.decode_first_stage(z_text, keypoints_mask)
        gloss_outs = self.gloss_decode(enc_outs, gloss[:, :-1], gloss_mask[:, :-1])

        return z_text, z_pose, generated, gloss_outs
        
    def get_inputs(self, batch):
        text = batch["text"]
        gloss = batch["gloss"]
        keypoints = batch["keypoints"]
        frame_lengths = batch["frame_lengths"]

        text = strings_to_indices(text, self.text_vocab)
        gloss = strings_to_indices(gloss, self.gloss_vocab)
        
        text = [torch.tensor(t) for t in text]
        gloss = [torch.tensor(g) for g in gloss]

        text = pad_sequence(text, batch_first=True, padding_value=self.text_vocab["<pad>"])
        gloss = pad_sequence(gloss, batch_first=True, padding_value=self.gloss_vocab["<pad>"])
        keypoints = pad_sequence(keypoints, batch_first=True, padding_value=0.)
        
        text_mask = (text != self.text_vocab["<pad>"])
        gloss_mask = (gloss != self.gloss_vocab["<pad>"])
        keypoints_mask = create_mask(frame_lengths, device=self.device)
        
        # Map tensor to device
        text, gloss, keypoints = map(lambda tensor: tensor.to(self.device), [text, gloss, keypoints])
        text_mask, gloss_mask, keypoints_mask = map(lambda tensor: tensor.to(self.device), [text_mask, gloss_mask, keypoints_mask])

        return (text, text_mask), (gloss, gloss_mask), (keypoints, keypoints_mask)

    def share_step(self, text, text_mask, gloss, gloss_mask, keypoints, keypoints_mask, split="train"):
        z_text, z_pose, generated, gloss_outs = self(text, text_mask, rearrange(keypoints, "b f v c -> b f (v c)"), 
                                                     keypoints_mask, gloss, gloss_mask)
        
        pose_loss = F.mse_loss(generated[keypoints_mask].contiguous(), keypoints[keypoints_mask].contiguous())
        gloss_loss = F.cross_entropy(gloss_outs.reshape(-1, gloss_outs.size(-1)), 
                                     gloss[:, 1:].reshape(-1), 
                                     ignore_index=self.gloss_vocab["<pad>"], label_smoothing=self.label_smoothing)

        loss = pose_loss + self.gloss_loss_weight * gloss_loss
        
        log_dict = {
            f"{split}/pose_loss": pose_loss.detach().mean(),
            f"{split}/gloss_loss": gloss_loss.detach().mean()
        }

        return loss, log_dict

    def training_step(self, batch, batch_idx):
        (text, text_mask), (gloss, gloss_mask), (keypoints, keypoints_mask) = self.get_inputs(batch)
        loss, log_dict = self.share_step(text, text_mask, gloss, gloss_mask, keypoints, keypoints_mask, split="train")

        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=text.shape[0])
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=text.shape[0])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        (text, text_mask), (gloss, gloss_mask), (keypoints, keypoints_mask) = self.get_inputs(batch)
        loss, log_dict = self.share_step(text, text_mask, gloss, gloss_mask, keypoints, keypoints_mask, split="valid")

        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=text.shape[0])
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=text.shape[0])
        
    def configure_optimizers(self):
        lr = self.learning_rate

        params = list(self.text_emb.parameters()) + list(self.gloss_emb.parameters()) + \
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
            list(self.latent_model.parameters()) + list(self.gloss_decoder.parameters()) + \
            list(self.gloss_outs.parameters())

        if self.first_stage_model:
            params = params + list(self.first_stage_model.parameters())

        optim = torch.optim.Adam(params, lr=lr)
        
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

        (text, text_mask), (gloss, gloss_mask), (keypoints, keypoints_mask) = self.get_inputs(batch)

        batch_size, nframes, _, _ = keypoints.shape

        enc_outs = self.encode(text, text_mask)
        time_qeuries = torch.zeros(batch_size, nframes, self.d_model, device=self.device)
        dec_outs = self.decode(enc_outs, time_qeuries, keypoints_mask)

        z_text = self.latent_model(dec_outs)

        generated = self.decode_first_stage(z_text, keypoints_mask)

        reference = keypoints

        reference, generated = list(map(partial(scaling_keypoints, 
                            width=return_scale_resoution()[0], height=return_scale_resoution()[1]), [reference, generated]))
        reference, generated = list(map(partial(center_keypoints, joint_idx=1), [reference, generated]))

        log["reference"] = reference
        log["generated"] = generated
        log["text"] = text

        return log 


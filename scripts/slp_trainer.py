from concurrent.futures import process
import os

import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data import load_data
from render import save_sign_video, save_sign_video_batch
from utils import noised, postprocess


class SignLanguageProductionTrainer(pl.LightningModule):
    def __init__(
        self, 
        dataset_type,
        min_seq_len,
        num_save,
        train_path,
        valid_path,
        test_path,
        batch_size,
        num_workers,
        lr,
        model,
        tokenizer,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        # dataset related parameters
        self.dataset_type = dataset_type
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        self.min_seq_len = min_seq_len

        # Training related paramters
        self.num_save = num_save
        self.batch_size = batch_size
        self.num_worker = num_workers
        self.lr = lr
        
        assert model != None, 'SLP model must be defined.'
        # assert tokenizer != None, 'Tokenizer must be defined.'
        
        self.model = model
        self.tokenizer = tokenizer

    def setup(self, stage):
        # load dataset
        self.trainset, self.validset, self.testset = \
            load_data(
                self.dataset_type, 
                self.train_path, 
                self.valid_path, 
                self.test_path, 
                seq_len = -1, 
                min_seq_len = self.min_seq_len
            )
        
        print(f'[INFO] {self.dataset_type} dataset loaded with sequence length {self.min_seq_len}.')

        if self.dataset_type == 'how2sign':
            self.S = 3
        else:
            self.S = 1.5

    def _common_step(self, batch, stage):
        raise NotImplementedError

    def _common_epoch_end(self, outputs, stage):
        H, W = 256, 256
        S = self.S

        output = outputs[0] # select only one example
    
        origin = output['origin']
        generated = output['generated']
        text = output['text']
        name = output['name']
        
        ans_toks = output['ans_toks']
        gen_toks = output['gen_toks']

        # display results
        print('============================================================')
        print(f'[INFO] Sample token outputs at epoch {self.current_epoch}')
        print(f'\nAnswer joint token: {ans_toks}')
        print(f'\nGenerated joint token: {gen_toks}')
        print('============================================================')
        
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

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, 'tr')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, 'val')

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, 'val')

    def configure_optimizers(self):
        raise NotImplementedError

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
            batch_size = 1, 
            shuffle = False, 
            num_workers = self.num_worker, 
            collate_fn = self._collate_fn
        )

    def _collate_fn(self, batch):
        id_list, text_list, joint_list = [], [], []
        joint_input_ids_list, joint_pad_mask_list, joint_input_logits_list = [], [], []
        
        sorted_batch = sorted(batch, key = lambda x: x['frame_len'], reverse = True)
        for data in sorted_batch:
            id_list.append(data['id'])
            text_list.append(data['text'])
            joint_list.append(data['joint_feats'])
            joint_input_ids_list.append(data['joint_input_ids'])
            joint_pad_mask_list.append(data['joint_pad_mask'])
            joint_input_logits_list.append(data['joint_input_logits'])

        return {
            'id': id_list,
            'text': text_list,
            'joints': joint_list,
            'joint_input_ids': joint_input_ids_list,
            'joint_pad_mask': joint_pad_mask_list,
            'joint_input_logits': joint_input_logits_list
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

        

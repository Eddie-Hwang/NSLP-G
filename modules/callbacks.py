import os
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
from modules.constant import *
from modules.helpers import instantiate_from_config


class KeypointsLogger(Callback):
    def __init__(
        self, 
        max_keypoints, 
        save_and_sample_every=10, 
        batch_freq=5, 
        renderer_config=None,
    ):
        self.save_and_sample_every = save_and_sample_every
        self.batch_freq = batch_freq
        self.max_keypoints = max_keypoints

        self.renderer = instantiate_from_config(renderer_config)
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]

    def check_frequency(self, check_idx):
        if ((check_idx in self.log_steps)) and (check_idx > 0):
            return True
        return False

    def log_local(
        self, 
        save_dir, 
        split, 
        reference,
        generated, 
        text,
        global_step, 
        current_epoch, 
        batch_idx
    ):
        root = os.path.join(save_dir, "keypoints", split)
        file_name = f"gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.gif"
        # file_name = f"gs-{global_step:06}_e-{current_epoch:06}.gif"
        path = os.path.join(root, file_name)
        os.makedirs(os.path.split(path)[0], exist_ok=True)

        self.renderer.save_animation(reference=reference, generated=generated, text=text, save_path=path)

    def log_keypoints(self, pl_module, batch, batch_idx, split="train", **kwargs):
        if not(pl_module.current_epoch % self.save_and_sample_every == 0):
            return
        
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_keypoints") and
                callable(pl_module.log_keypoints) and
                self.max_keypoints > 0):
        
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            
            with torch.no_grad():
                outputs = pl_module.log_keypoints(batch, **kwargs)

            text = outputs["text"]
            generated = outputs["generated"]
            reference = outputs["reference"]
            
            N = min(len(text), self.max_keypoints)
            selected_samples = torch.randperm(len(text))[:N] # Select N random samples from the batch
            txt = [text[idx] for idx in selected_samples.tolist()]
            gen = generated[selected_samples].detach().cpu()
            ref = reference[selected_samples].detach().cpu()

            for t, g, r in zip(txt, gen, ref):
                self.log_local(pl_module.logger.save_dir, split, r, g, t, 
                        pl_module.global_step, pl_module.current_epoch, batch_idx)

    def log_data(self, pl_module, batch, batch_idx, split="train", **kwargs):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            outputs = pl_module.log_keypoints(batch, **kwargs)

        root = os.path.join(pl_module.logger.save_dir, split)
        text_list, gloss_list, keypoint_list, mask_list = \
            outputs["text"], outputs["gloss"], outputs["generated"], outputs["mask"]
        for i, (t, g, k, m) in enumerate(zip(text_list, gloss_list, keypoint_list, mask_list)):
            file_name = f"id-{i}_gs-{pl_module.global_step}_b-{batch_idx:06}.torch"
            path = os.path.join(root, file_name)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            dict_out = {
                "text": t,
                "gloss": g,
                "keypont": k[:m.sum()],
            }
            torch.save(dict_out, path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_keypoints(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_keypoints(pl_module, batch, batch_idx, split="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # self.log_keypoints(pl_module, batch, batch_idx, split="test")
        self.log_data(pl_module, batch, batch_idx, split="test")
        

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0:
            if pl_module.global_step != 0:
                print("[INFO] Summoning checkpoint.")
                ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

    def on_train_start(self, trainer, pl_module):
        if self.resume is not None:
            return
        
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # print("Project config")
            # print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            # print("Lightning config")
            # print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))
            

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
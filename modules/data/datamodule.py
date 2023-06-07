import pytorch_lightning as pl
from torch.utils.data import DataLoader
from modules.normalize import preprocess_keypoints
from modules.helpers import *


def custom_collate(batch):
    text_list = [elem.text for elem in batch]
    gloss_list = [elem.gloss for elem in batch]
    
    keypoints_tensor_list = [elem.keypoint for elem in batch]
    keypoints_tensor_list = [preprocess_keypoints(keypoint) for keypoint in keypoints_tensor_list]
    # keypoints_tensor_list = [center_keypoints(keypoint) for keypoint in keypoints_tensor_list]
    # keypoints_tensor_list = [remove_noisy_frames(keypoint) for keypoint in keypoints_tensor_list]
    # keypoints_tensor_list = [normalize_skeleton_3d(keypoint) for keypoint in keypoints_tensor_list]
    # keypoints_tensor_list = [normalize(keypoint) for keypoint in keypoints_tensor_list]

    frame_length_list = [keypoint.shape[0] for keypoint in keypoints_tensor_list]
    
    padded_keypoints_tensor = torch.nn.utils.rnn.pad_sequence(keypoints_tensor_list, batch_first=True, padding_value=0.)
    
    return {
        "text": text_list,
        "gloss": gloss_list,
        "keypoints": padded_keypoints_tensor,
        "frame_lengths": frame_length_list
    }


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, num_workers=None, pin_memory=False):
        super().__init__()

        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.pin_memory = pin_memory

        if train is not None:
            self.dataset_configs['train'] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs['valid'] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = self._test_dataloader

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )
        
    def _train_dataloader(self):
        return DataLoader(
            dataset=self.datasets['train'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
            pin_memory=self.pin_memory,
            shuffle=True
        )

    def _val_dataloader(self):
        return DataLoader(
            dataset=self.datasets['valid'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    def _test_dataloader(self):
        return DataLoader(
            dataset=self.datasets['test'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=custom_collate,
            pin_memory=self.pin_memory,
            shuffle=False
        ) 
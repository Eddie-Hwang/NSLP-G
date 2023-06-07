from collections import namedtuple
from einops import rearrange
from torch.utils.data import Dataset
import torch
import gzip
import pickle


def get_data(path):
    with open(path, "r") as file:
        return [line.strip() for line in file]


def string_to_keypoint_tensor(data, trg_size=150+1):
    frame = [float(val) for val in data.split()]
    keypoint_tensor = torch.tensor(frame)
    keypoint_tensor = keypoint_tensor.view(-1, trg_size)
    counter_tensor = keypoint_tensor[:, -1]
    keypoint_tensor = keypoint_tensor[:, :-1]
    keypoint_tensor = rearrange(keypoint_tensor, "f (j d) -> f j d", d=3)

    return keypoint_tensor, counter_tensor


class Phoenix2014T(Dataset):
    """
    Phoenix2014T dataset for sign language recognition.

    This dataset class provides the text, gloss, keypoint, and counter data for each sample in the dataset.

    Note that the keypoints does not include landmarks.
    """

    def __init__(
        self,
        text_file_path,
        gloss_file_path,
        keypoint_file_path,
    ):
        """
        Initialize the dataset with text, gloss, and keypoint file paths.

        Args:
            text_file_path (str): Path to the text data file.
            gloss_file_path (str): Path to the gloss data file.
            keypoint_file_path (str): Path to the keypoint data file.
        """
        super().__init__()

        self.text_data = get_data(text_file_path)
        self.gloss_data = get_data(gloss_file_path)
        self.keypoint_data = get_data(keypoint_file_path)
        self.data = list(map(string_to_keypoint_tensor, self.keypoint_data))

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.text_data)
    
    def __getitem__(self, index):
        """
        Get a dataset sample given an index.

        Args:
            index (int): The index of the desired sample.

        Returns:
            Sample (namedtuple): A namedtuple containing the text, gloss, keypoint, and counter data for the sample.
        """
        text = self.text_data[index]
        gloss = self.gloss_data[index]
        keypoints, counter = self.data[index]

        Sample = namedtuple('Sample', ['text', 'gloss', 'keypoint', 'counter'])
        return Sample(text, gloss, keypoints, counter)

    

class Phoenix2014TDepreciated(Dataset):
    def __init__(self, file_path):
        super().__init__()

        self.data = self.load_data(file_path)        

    def load_data(self, fpath):
        with gzip.open(fpath, "rb") as f:
            data = pickle.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        text = data["text"]
        gloss = data["gloss"]
        keypoint = torch.tensor(data["sign"])
        keypoint = rearrange(keypoint, "f (v c) -> f v c", c=2)
        keypoint = keypoint[:, ]
        
        Sample = namedtuple("Sample", ["text", "gloss", "keypoint"])
        return Sample(text, gloss, keypoint)

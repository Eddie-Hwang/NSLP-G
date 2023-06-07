import torch
import importlib
import json
from collections import Counter


def create_mask(seq_lengths, device="cpu"):
    max_len = max(seq_lengths)
    mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device)[:, None]
    return mask.bool()


def add_noise(x, noise_std=0.5):
    noise = torch.randn_like(x)
    return x + noise * noise_std


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key "target" to instatiate.')
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_vocab(file_path, min_freq=1):
    data = get_data(file_path)
    vocab = build_vocab(data, min_freq)
    return vocab


def save_vocab(vocab, file_path):
    with open(file_path, "w") as f:
        json.dump(vocab, f, indent=4)


def load_vocab(file_path):
    with open(file_path, "r") as f:
        vocab = json.load(f)
    return vocab


def get_data(path):
    with open(path, "r") as file:
        return [line.strip() for line in file]
    

def build_vocab(data, min_freq):
    """
    Build a vocabulary from the given data.

    Args:
        data (list[str]): A list of strings, where each string represents a line of data.
        min_freq (int): The minimum frequency for a token to be included in the vocabulary.

    Returns:
        dict: A dictionary mapping tokens to their indices in the vocabulary.
    """
    tokens = [token for line in data for token in line.split()]
    token_counts = Counter(tokens)
    token_list = [token for token, count in token_counts.items() if count >= min_freq]

    # Add special tokens to the vocabulary
    special_tokens = ['<sos>', '<eos>', '<pad>']
    vocab = {token: idx for idx, token in enumerate(special_tokens + token_list)}

    return vocab


def strings_to_indices(strings, vocab):
    """
    Convert a list of strings to a list of lists of indices based on the given vocabulary.

    Args:
        strings (list[str]): The input list of strings to be converted.
        vocab (dict): The vocabulary mapping tokens to indices.

    Returns:
        list[list[int]]: A list of lists of indices representing the input strings.
    """
    indices_list = []
    for string in strings:
        tokens = string.split()
        indices = [vocab["<sos>"]] + [vocab[token] for token in tokens if token in vocab] + [vocab["<eos>"]]
        indices_list.append(indices)
    return indices_list


def indices_to_strings(indices_list, vocab, sos_token='<sos>', eos_token='<eos>', pad_token='<pad>'):
    """
    Convert a list of lists of indices to a list of strings based on the given vocabulary.

    Args:
        indices_list (list[list[int]]): The input list of lists of indices to be converted.
        vocab (dict): The vocabulary mapping tokens to indices.

    Returns:
        list[str]: A list of strings representing the input indices.
    """
    inv_vocab = {idx: token for token, idx in vocab.items()}
    special_indices = [vocab[token] for token in [sos_token, eos_token, pad_token]]
    strings = []
    for indices in indices_list:
        indices = indices.tolist()
        tokens = [inv_vocab[idx] for idx in indices if idx not in special_indices]
        string = ' '.join(tokens)
        strings.append(string)
    return strings
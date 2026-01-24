import numpy as np
import torch
import argparse
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split

from config import ModelConfig


class CorpusDataset(Dataset):
    def __init__(self, path):
        self.data = self._load_data(path=path)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _load_data(self, path):
        return np.load(path, allow_pickle=True)


def train_test_split(dataset, train_size):
    val_size = 1 - train_size
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [train_size, val_size], generator=generator)


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    Pads sequences to the same length within each batch
    """
    # Trimming the sequence length

    batch = [seq[:ModelConfig().context_size] for seq in batch]

    return pad_sequence(batch, batch_first=True, padding_value=0)


DATA_PATHS = {
    'spm': 'data/model_training/corpus_spm.npy',
    'pl': 'data/model_training/corpus_pl.npy',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train Polish Language Model')

    # Data mode
    parser.add_argument('--mode', type=str, choices=['spm', 'pl'], default='pl', help='Tokenizer mode: spm or pl')

    # Checkpointing
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default='training/models/checkpoints/', help='Directory to save checkpoints')

    # Hardware
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--precision', type=str, default='16-mixed', choices=['16-mixed', '32', 'bf16-mixed'], help='Training precision (default: 16-mixed)')

    return parser.parse_args()


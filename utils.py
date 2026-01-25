import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, random_split

from config import ModelConfig


class CorpusDataset(Dataset):
    def __init__(self, path, context_size=ModelConfig.context_size):
        self.context_size = context_size
        self.data = np.memmap(path, dtype=np.uint16, mode='r')

    def __getitem__(self, index):
        start = index * self.context_size
        end = start + self.context_size
        return torch.tensor(self.data[start:end], dtype=torch.long)

    def __len__(self):
        return len(self.data) // self.context_size



def train_test_split(dataset, train_size):
    val_size = 1 - train_size
    generator = torch.Generator().manual_seed(42)
    return random_split(dataset, [train_size, val_size], generator=generator)


def collate_fn(batch):
    return torch.stack(batch)


DATA_PATHS = {
    'spm': 'data/model_training/corpus_spm.bin',
    'pl': 'data/model_training/corpus_pl.bin',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train Polish Language Model')

    # For starting the proces from scratch
    parser.add_argument('--start_fresh', type=bool, default=False, help='Start fresh process - download and process data')

    # Data mode
    parser.add_argument('--mode', type=str, choices=['spm', 'pl'], default='pl', help='Tokenizer mode: spm or pl')
    parser.add_argument('--max_epochs', type=int, default=4, help='Maximum number of training epochs (default: 10)')

    # Checkpointing
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default='training/models/checkpoints/', help='Directory to save checkpoints')

    # Hardware
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--precision', type=str, default='16-mixed', choices=['16-mixed', '32', 'bf16-mixed'], help='Training precision (default: 16-mixed)')

    args = parser.parse_args()
    args.data_path = DATA_PATHS[args.mode]
    return args


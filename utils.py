import os
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, random_split
import gdown

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

# Google Drive file IDs for pre-tokenized corpora
GDRIVE_FILE_IDS = {
    'pl': '1NH5cU2IlzFAsH2QAljsZDpeEGILLGvZg',
    'spm': '1Ngap9Q5kZJOTiHmGDs7upmRCGS3KAB8q',
}


def download_corpus(mode: str, output_dir: str = 'data/model_training'):
    """Download pre-tokenized corpus from Google Drive if not exists."""


    output_path = DATA_PATHS[mode]

    if os.path.exists(output_path):
        print(f"Corpus already exists: {output_path}")
        return output_path

    os.makedirs(output_dir, exist_ok=True)

    file_id = GDRIVE_FILE_IDS[mode]
    url = f'https://drive.google.com/uc?id={file_id}'

    print(f"Downloading {mode} corpus from Google Drive...")
    gdown.download(url, output_path, quiet=False)
    print(f"Downloaded to: {output_path}")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description='Train Polish Language Model')

    # For starting the proces from scratch
    parser.add_argument('--start_fresh', type=bool, default=False, help='Start fresh process - download and process data')

    # Data mode
    parser.add_argument('--mode', type=str, choices=['spm', 'pl'], default='pl', help='Tokenizer mode: spm or pl')
    parser.add_argument('--max_epochs', type=int, default=4, help='Maximum number of training epochs (default: 10)')

    # Checkpointing
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints (default: models/{mode}/)')

    # Hardware
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--precision', type=str, default='16-mixed', choices=['16-mixed', '32', 'bf16-mixed'], help='Training precision (default: 16-mixed)')

    args = parser.parse_args()
    args.data_path = DATA_PATHS[args.mode]
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f'models/{args.mode.upper()}/'
    return args


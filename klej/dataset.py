import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List
from sklearn.preprocessing import LabelEncoder

from .tasks import TASKS, TaskConfig


class TokenizerWrapper:
    """Unified interface for both tokenizers."""

    def __init__(self, tokenizer, mode: str):
        self.tokenizer = tokenizer
        self.mode = mode
        self._pad_id = 0  # Both tokenizers use 0 for padding

    def encode(self, text: str) -> List[int]:
        if self.mode == 'pl':
            return self.tokenizer.tokens_to_ids(text)
        else:  # spm
            return self.tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        if self.mode == 'pl':
            return self.tokenizer.ids_to_tokens(ids)
        else:  # spm
            return self.tokenizer.decode(ids)

    @property
    def pad_id(self) -> int:
        return self._pad_id


class KLEJDataset(Dataset):
    """
    Dataset for KLEJ benchmark tasks.

    Handles:
    - Loading TSV files
    - Tokenization with either PL or SPM tokenizer
    - Label encoding for classification tasks
    - Padding/truncation to max_length
    """

    def __init__(
        self,
        task_name: str,
        split: str,  # 'train', 'dev', 'test'
        tokenizer: TokenizerWrapper,
        max_length: int = 512,
        data_dir: str = str(Path(__file__).parent.parent / 'data' / 'klej' / 'KLEJ'),
    ):
        self.task_config = TASKS[task_name]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        # Load data
        task_dir = os.path.join(data_dir, self.task_config.dir_name)
        if split == 'test':
            file_path = os.path.join(task_dir, 'test_features.tsv')
        else:
            file_path = os.path.join(task_dir, f'{split}.tsv')

        self.df = pd.read_csv(file_path, sep='\t')

        # Setup label encoder for classification tasks
        self.label_encoder = None
        if self.task_config.task_type == 'classification' and split != 'test':
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(
                self.df[self.task_config.target_column].values
            )
            self.num_labels = len(self.label_encoder.classes_)
        elif self.task_config.task_type == 'regression' and split != 'test':
            self.labels = self.df[self.task_config.target_column].values.astype(float)
            self.num_labels = 1
        else:
            self.labels = None
            self.num_labels = self.task_config.num_labels

        # Precompute tokenized inputs
        self.input_ids = []
        self.attention_masks = []
        self._preprocess()

    def _preprocess(self):
        """Tokenize and pad all examples."""
        for idx in range(len(self.df)):
            # Combine text columns with [SEP] token for pair tasks
            texts = []
            for col in self.task_config.text_columns:
                text = str(self.df.iloc[idx][col])
                texts.append(text)

            # Join with separator for pair tasks
            combined_text = ' [SEP] '.join(texts)

            # Tokenize
            tokens = self.tokenizer.encode(combined_text)

            # Truncate if needed (leave room for nothing special - decoder model)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(tokens)

            # Pad to max_length
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [self.tokenizer.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

            self.input_ids.append(tokens)
            self.attention_masks.append(attention_mask)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }

        if self.labels is not None:
            if self.task_config.task_type == 'classification':
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            else:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item

    def get_label_encoder(self):
        """Return label encoder for inverse transform during evaluation."""
        return self.label_encoder


def load_klej_datasets(
    task_name: str,
    tokenizer: TokenizerWrapper,
    max_length: int = 512,
    data_dir: str = str(Path(__file__).parent.parent / 'data' / 'klej' / 'KLEJ'),
):
    """
    Load train, dev, and test datasets for a KLEJ task.

    Returns:
        dict with 'train', 'dev' (if exists), 'test' datasets
    """
    task_config = TASKS[task_name]

    datasets = {
        'train': KLEJDataset(task_name, 'train', tokenizer, max_length, data_dir),
    }

    if task_config.has_dev:
        datasets['dev'] = KLEJDataset(task_name, 'dev', tokenizer, max_length, data_dir)
        # Copy label encoder from train to dev
        datasets['dev'].label_encoder = datasets['train'].label_encoder

    datasets['test'] = KLEJDataset(task_name, 'test', tokenizer, max_length, data_dir)

    return datasets
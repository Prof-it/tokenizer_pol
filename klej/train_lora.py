import os
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gdown
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from config import ModelConfig, LoRAConfig, LoRATrainingsConfig
from klej.tasks import TASKS
from klej.dataset import TokenizerWrapper, load_klej_datasets
from klej.classifier import KLEJClassifier

config = ModelConfig()
lora_config = LoRAConfig()
lora_training = LoRATrainingsConfig()


CHECKPOINT_GDRIVE_IDS = {
    'pl':  '1sY8LhHyPKE2m2FzOcjHP466bbLgNyX8W',
    'spm': '1wahTrHkbuW4u424Z8rTMEOu_KWcs6Kys',
}


def ensure_pl_tokenizer():
    from huggingface_hub import snapshot_download
    tokenizer_dir = Path(__file__).parent.parent / 'tokenizer'
    if not tokenizer_dir.exists() or not any(tokenizer_dir.iterdir()):
        print("Polish tokenizer not found, downloading from HuggingFace...")
        snapshot_download(
            repo_id="rafal-adamczyk/polish-morphological-tokenizer",
            local_dir=str(tokenizer_dir),
            token=True,
        )


def _is_valid_checkpoint(path: Path) -> bool:
    import zipfile
    try:
        return zipfile.is_zipfile(str(path))
    except Exception:
        return False


def ensure_checkpoint(mode: str, project_root: Path) -> str:
    """Download pretrained checkpoint from Google Drive if not present or corrupted."""
    folder = 'PL' if mode == 'pl' else 'SPM'
    checkpoint_path = project_root / 'models' / folder / 'last.ckpt'

    if checkpoint_path.exists() and not _is_valid_checkpoint(checkpoint_path):
        print(f"Corrupt checkpoint detected at {checkpoint_path}, removing...")
        checkpoint_path.unlink()

    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        file_id = CHECKPOINT_GDRIVE_IDS[mode]
        print(f"Downloading {mode} checkpoint from Google Drive...")
        gdown.download(id=file_id, output=str(checkpoint_path), quiet=False, resume=True)

        if not _is_valid_checkpoint(checkpoint_path):
            checkpoint_path.unlink()
            raise RuntimeError(
                f"Downloaded checkpoint for '{mode}' is not a valid PyTorch file. "
                "The Google Drive link may require additional permissions or the upload may be incomplete."
            )
        print(f"Checkpoint saved to {checkpoint_path}")

    return str(checkpoint_path)


def load_tokenizer(mode: str):
    """Load tokenizer based on mode."""
    if mode == 'pl':
        # Add tokenizer directory to path
        tokenizer_dir = Path(__file__).parent.parent / 'tokenizer'
        sys.path.insert(0, str(tokenizer_dir))
        from tokenizer import PolishToKenizer

        tokenizer = PolishToKenizer()
        tokenizer.load_vocab(tokenizer_dir / 'vocab.json')
        return TokenizerWrapper(tokenizer, mode='pl')
    else:
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(Path(__file__).parent.parent / 'data/spm/baseline_tokenizer.model'))
        return TokenizerWrapper(tokenizer, mode='spm')


def train_task(task_name:str, mode:str):

    torch.set_float32_matmul_precision('medium')

    print(f"\n{'='*60}")
    print(f"Training {task_name} with {mode} tokenizer")
    print(f"{'='*60}\n")

    project_root = Path(__file__).parent.parent

    checkpoint_path = ensure_checkpoint(mode, project_root)

    output_dir = str(project_root / 'models' / 'klej')

    tokenizer = load_tokenizer(mode)
    datasets = load_klej_datasets(task_name=task_name, tokenizer=tokenizer, max_length=config.context_size)


    train_loader = DataLoader(
        datasets['train'],
        batch_size=lora_training.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = None
    if 'dev' in datasets:
        val_loader = DataLoader(
            datasets['dev'],
            batch_size=lora_training.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

    model = KLEJClassifier(
        task_name=task_name,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        context_size=config.context_size,
        d_ff=config.d_ff,
        dropout=config.dropout,
        num_labels=datasets['train'].num_labels,
        learning_rate=lora_config.learning_rate,
        lora_rank=lora_config.rank,
    )

    model.load_pretrained_weights(checkpoint_path)

    # Setup output directory
    task_output_dir = os.path.join(output_dir, mode, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=task_output_dir,
        filename='best-{epoch:02d}-{val_loss:.3f}' if val_loader else 'best-{epoch:02d}-{train_loss:.3f}',
        monitor='val_loss' if val_loader else 'train_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )

    logger = WandbLogger(
        project='polish-morph-bpe',
        name=f'klej-{task_name}-{mode}',
        tags=['lora-finetune', f'task:{task_name}', f'tokenizer:{mode}', f'rank:{lora_config.rank}'],
    )

    # Log hyperparameters
    logger.experiment.config.update({
        'task': task_name,
        'tokenizer_mode': mode,
        'lora_rank': lora_config.rank,
        'learning_rate': lora_config.learning_rate,
        'batch_size': lora_training.batch_size,
        'max_length': config.context_size,
        'max_epochs': lora_training.max_epochs,
    })

    # Trainer
    trainer = L.Trainer(
        max_epochs=lora_training.max_epochs,
        accelerator='auto',
        devices=1,
        precision='16-mixed',
        callbacks=[
            checkpoint_callback,
            RichModelSummary(max_depth=2),
            LearningRateMonitor(logging_interval='step')
        ],
        logger=logger,
        gradient_clip_val=1.0,
    )

    # Train
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print(f"\nTraining complete! Model saved to {task_output_dir}")

    # Return best model path for evaluation
    return checkpoint_callback.best_model_path


def ensure_klej_data():
    klej_dir = Path(__file__).parent.parent / 'data' / 'klej' / 'KLEJ'
    if not klej_dir.exists() or not any(klej_dir.iterdir()):
        print("KLEJ data not found, downloading...")
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / 'download_klej.py')],
            cwd=str(Path(__file__).parent),
            check=True,
        )


def main():

    ensure_klej_data()

    if lora_training.mode == 'pl':
        ensure_pl_tokenizer()

    results = {}

    tasks = list(TASKS.keys())

    for task in tasks:
        print(f"\nTraining {task}")
        best_ckpt = train_task(
            task_name=task,
            mode=lora_training.mode,
        )
        results[task] = {'status': 'success', 'checkpoint': best_ckpt}

    # Print summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")

    for task_name, result in results.items():
        status = '✓' if result['status'] == 'success' else '✗'
        print(f"  {status} {task_name}: {result['status']}")

if __name__ == '__main__':
    main()

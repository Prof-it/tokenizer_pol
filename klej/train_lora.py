import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

    if mode == 'pl':
        checkpoint_path = str(project_root / 'models' / 'PL' / 'last.ckpt')
    else:
        checkpoint_path = str(project_root / 'models' / 'SPM' / 'last.ckpt')

    output_dir = str(project_root / 'models' / 'klej')

    tokenizer = load_tokenizer(mode)
    datasets = load_klej_datasets(task_name=task_name, tokenizer=tokenizer, max_length=config.context_size)


    train_loader = DataLoader(
        datasets['train'],
        batch_size=lora_training.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )

    val_loader = None
    if 'dev' in datasets:
        val_loader = DataLoader(
            datasets['dev'],
            batch_size=lora_training.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
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

    # Check for wandb run resume
    # wandb_run_id = None
    # wandb_id_file = os.path.join(task_output_dir, 'wandb_run_id.txt')
    #
    # if os.path.exists(wandb_id_file):
    #     with open(wandb_id_file, 'r') as f:
    #         wandb_run_id = f.read().strip()
    #     print(f"Resuming W&B run: {wandb_run_id}")

    logger = WandbLogger(
        project='polish-morph-bpe',
        name=f'klej-{task_name}-{mode}',
        tags=['lora-finetune', f'task:{task_name}', f'tokenizer:{mode}', f'rank:{lora_config.rank}'],
        # id=wandb_run_id,
        # resume='must' if wandb_run_id else None,
    )

    # Save run ID for future resume
    # if not wandb_run_id:
    #     with open(wandb_id_file, 'w') as f:
    #         f.write(logger.experiment.id)

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


def main():

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

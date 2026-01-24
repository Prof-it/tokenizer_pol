import os
import torch
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, LearningRateMonitor

from config import ModelConfig, TrainingConfig
from lightning_lm import LMTraining
from utils import *
from build_dataset import data_pipeline


def train():
    args = parse_args()

    model_config = ModelConfig()
    training_config = TrainingConfig()

    torch.set_float32_matmul_precision('medium')

    # Setup dataset and dataloader
    dataset = CorpusDataset(args.data_path)

    # Split data 4:1
    train_data, val_data = train_test_split(dataset, train_size=0.8)

    # Training
    train_dataloader = DataLoader(
        train_data,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Validation
    val_dataloader = DataLoader(
        val_data,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Initialize model
    model = LMTraining(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        context_size=model_config.context_size,
        d_ff=model_config.d_ff,
        dropout=model_config.dropout,
        learning_rate=model_config.learning_rate,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='model-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min',
        save_last=True
    )

    # Setup logger
    logger = WandbLogger(project='polish-morph-bpe')

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy='ddp' if args.devices > 1 else "auto",
        precision=args.precision,
        gradient_clip_val=1.0,
        callbacks=[
            checkpoint_callback,
            RichModelSummary(max_depth=2),
            # RichProgressBar(),
            LearningRateMonitor(logging_interval='step')
        ],
        enable_checkpointing=True,
    )

    # Resume from checkpoint
    ckpt_path = None
    if args.resume_from:
        ckpt_path = args.resume_from
        print(f"Resuming from checkpoint: {ckpt_path}")
    elif os.path.exists(os.path.join(args.checkpoint_dir, 'last.ckpt')):
        ckpt_path = os.path.join(args.checkpoint_dir, 'last.ckpt')
        print(f"Resuming from last checkpoint: {ckpt_path}")

    # Start training
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    os.makedirs("data/model_training", exist_ok=True)
    train()

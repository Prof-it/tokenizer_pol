import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Optional

from language_model import LanguageModel
from .lora import apply_lora, get_lora_params, count_parameters
from .tasks import TASKS


class KLEJClassifier(L.LightningModule):
    """
    Architecture:
    - base model from pre-training
    - Take the last non-padded token's hidden state
    - Classification head: LayerNorm → Dropout → Linear → num_labels
    """

    def __init__(
        self,
        task_name: str,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        context_size: int,
        d_ff: int,
        dropout: float,
        num_labels: Optional[int],
        learning_rate: float,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        classifier_dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # name, dirname, text labels etc
        self.task_config = TASKS[task_name]
        self.task_type = self.task_config.task_type
        self.num_labels = num_labels

        # Store LoRA config
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self._lora_applied = False

        # inherited base model
        self.model = LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            context_size=context_size,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Classification head added on top of bsae LM
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(classifier_dropout),
            nn.Linear(d_model, self.num_labels),
        )

        self.learning_rate = learning_rate
        self.d_model = d_model

    def load_pretrained_weights(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Handle Lightning checkpoint format
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (from LMTraining wrapper)
            state_dict = {
                k.replace('model.', ''): v
                for k, v in state_dict.items()
                if k.startswith('model.')
            }
        else:
            state_dict = checkpoint

        # Load weights into base model
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        print(f"Loaded pretrained weights from {checkpoint_path}")
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

        # Freeze all base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # add LORA
        if not self._lora_applied:
            self.model = apply_lora(
                self.model,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
            self._lora_applied = True

        # Print parameter counts
        param_counts = count_parameters(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        trainable = param_counts['lora'] + classifier_params
        print(f"Model parameters: {param_counts}")
        print(f"Total: {total_params + classifier_params:,} | Trainable: {trainable:,} ({100*trainable/(total_params+classifier_params):.2f}%)")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        # Get hidden states from base model
        # embed → pos_enc → transformer layers → layer_norm → LORA here
        # stop before output projection

        x = self.model.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.model.positional_encoding(x)

        for layer in self.model.transformer_layers:
            x = layer(x)

        x = self.model.layer_norm(x)  # (batch, seq_len, d_model)

        # Get last non-padded token for each sequence
        # attention_mask: 1 for real tokens, 0 for padding
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(x.size(0), device=x.device)

        last_hidden = x[batch_indices, seq_lengths]  # (batch, d_model)

        # Classification head
        logits = self.classifier(last_hidden)  # (batch, num_labels)

        return logits

    def _compute_loss(self, logits, labels):
        """Compute loss based on task type."""
        if self.task_type == 'classification':
            return F.cross_entropy(logits, labels)
        else:  # regression
            return F.mse_loss(logits.squeeze(-1), labels)

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self._compute_loss(logits, batch['labels'])

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self._compute_loss(logits, batch['labels'])

        # Compute metrics
        if self.task_type == 'classification':
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == batch['labels']).float().mean()
            self.log('val_acc', acc, prog_bar=True)
        else:
            # For regression, log MAE
            mae = torch.abs(logits.squeeze(-1) - batch['labels']).mean()
            self.log('val_mae', mae, prog_bar=True)

        self.log('val_loss', loss, prog_bar=True)
        return {'loss': loss, 'logits': logits, 'labels': batch['labels']}

    def predict_step(self, batch, batch_idx):
        """Generate predictions for test set."""
        logits = self(batch['input_ids'], batch['attention_mask'])

        if self.task_type == 'classification':
            preds = torch.argmax(logits, dim=-1)
        else:
            preds = logits.squeeze(-1)

        return preds

    def configure_optimizers(self):
        # Only optimize LoRA parameters and classifier
        lora_params = get_lora_params(self.model)
        classifier_params = list(self.classifier.parameters())

        optimizer = torch.optim.AdamW(
            lora_params + classifier_params,
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        # Linear warmup + cosine decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.learning_rate * 0.1,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
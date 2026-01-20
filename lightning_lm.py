import torch
import torch.nn.functional as F
import lightning as L

from language_model import LanguageModel

class LMTraining(L.LightningModule):
    def __init__(self,
                 vocab_size,
                 d_model,
                 num_heads,
                 num_layers,
                 context_size,
                 d_ff,
                 dropout,
                 learning_rate=3e-4
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.model = LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            context_size=context_size,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Up to the last token
        inputs = batch[:, :-1]
        # Shift to all but first token
        targets = batch[:, 1:]

        # Forward pass
        logits = self(inputs)  # [batch_size, seq_len-1, vocab_size]

        loss = F.cross_entropy(
            input=logits.reshape(-1, logits.size(-1)),  # [batch*seq, vocab_size]
            target=targets.reshape(-1),  # [batch*seq]
            ignore_index = 0 # ignore padding
        )

        perplexity = torch.exp(loss)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_perplexity', perplexity, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = self(inputs)

            loss = F.cross_entropy(
                input=logits.reshape(-1, logits.size(-1)),
                target=targets.reshape(-1),
                ignore_index=0
            )

            perplexity = torch.exp(loss)

            self.log('val_loss', loss, prog_bar=True)
            self.log('val_perplexity', perplexity, prog_bar=True)

            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
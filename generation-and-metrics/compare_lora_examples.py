"""
Requires trained LoRA checkpoints in models/klej/{pl,spm}/{task}/best-*.ckpt
Requires KLEJ data in data/klej/KLEJ/
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from config import ModelConfig
from klej.tasks import TASKS
from klej.dataset import KLEJDataset
from klej.classifier import KLEJClassifier
from klej.train_lora import load_tokenizer

CHECKPOINT_DIR = Path(__file__).parent / 'models' / 'klej'
BATCH_SIZE = 32
MAX_EXAMPLES_PER_TASK = 10


def find_best_ckpt(mode: str, task_name: str) -> str | None:
    task_dir = CHECKPOINT_DIR / mode / task_name
    candidates = sorted(task_dir.glob('best-*.ckpt'))
    if not candidates:
        return None
    return str(candidates[0])


def load_classifier(task_name: str, checkpoint_path: str) -> KLEJClassifier:
    cfg = ModelConfig()
    model = KLEJClassifier.load_from_checkpoint(
        checkpoint_path,
        task_name=task_name,
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        context_size=cfg.context_size,
        d_ff=cfg.d_ff,
        map_location='cpu',
    )
    model.eval()
    return model


def run_inference(model: KLEJClassifier, dataset: KLEJDataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    task_type = TASKS[dataset.task_config.name].task_type
    all_preds = []

    model = model.to(device)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)

            if task_type == 'classification':
                preds = torch.argmax(logits, dim=-1)
            else:
                preds = logits.squeeze(-1)

            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)


def get_text_for_row(row, text_columns):
    return ' | '.join(str(row[col]) for col in text_columns)


def analyze_task(task_name: str, device: torch.device, pl_tok, spm_tok):
    task_cfg = TASKS[task_name]

    pl_ckpt = find_best_ckpt('pl', task_name)
    spm_ckpt = find_best_ckpt('spm', task_name)

    if not pl_ckpt or not spm_ckpt:
        missing = [m for m, c in [('pl', pl_ckpt), ('spm', spm_ckpt)] if not c]
        print(f"  [{task_name}] Skipping — no checkpoint for: {', '.join(missing)}")
        return

    if not task_cfg.has_dev:
        print(f"  [{task_name}] Skipping — no dev split available")
        return

    split = 'dev'

    pl_dataset = KLEJDataset(task_name, split, pl_tok)
    spm_dataset = KLEJDataset(task_name, split, spm_tok)

    if task_cfg.task_type == 'classification':
        spm_dataset.label_encoder = pl_dataset.label_encoder

    pl_model = load_classifier(task_name, pl_ckpt)
    pl_preds = run_inference(pl_model, pl_dataset, device)
    del pl_model

    spm_model = load_classifier(task_name, spm_ckpt)
    spm_preds = run_inference(spm_model, spm_dataset, device)
    del spm_model

    true_labels = np.array(pl_dataset.labels)

    if task_cfg.task_type == 'classification':
        pl_correct = pl_preds == true_labels
        spm_correct = spm_preds == true_labels
        pl_wins = np.where(pl_correct & ~spm_correct)[0]
        overall_pl_acc = pl_correct.mean()
        overall_spm_acc = spm_correct.mean()
    else:
        pl_err = np.abs(pl_preds - true_labels)
        spm_err = np.abs(spm_preds - true_labels)
        pl_wins = np.where((spm_err - pl_err) >= 0.5)[0]
        overall_pl_acc = overall_spm_acc = None

    print(f"\n{'=' * 70}")
    print(f"Task: {task_name}  |  split: {split}  |  metric: {task_cfg.metric}")
    if task_cfg.task_type == 'classification':
        print(f"PL  accuracy: {overall_pl_acc:.4f}   SPM accuracy: {overall_spm_acc:.4f}")
    print(f"Examples where PL correct, SPM wrong: {len(pl_wins)}")
    print('=' * 70)

    if len(pl_wins) == 0:
        print("  (none found)")
        return

    label_enc = pl_dataset.label_encoder if task_cfg.task_type == 'classification' else None

    for i, idx in enumerate(pl_wins[:MAX_EXAMPLES_PER_TASK]):
        row = pl_dataset.df.iloc[idx]
        text = get_text_for_row(row, task_cfg.text_columns)

        true_val = true_labels[idx]
        pl_val = pl_preds[idx]
        spm_val = spm_preds[idx]

        if label_enc is not None:
            true_str = label_enc.inverse_transform([int(true_val)])[0]
            pl_str = label_enc.inverse_transform([int(pl_val)])[0]
            spm_str = label_enc.inverse_transform([int(spm_val)])[0]
        else:
            true_str = f"{true_val:.2f}"
            pl_str = f"{pl_val:.2f}"
            spm_str = f"{spm_val:.2f}"

        print(f"\n  [{i + 1}] {text[:200]}")
        print(f"       true={true_str}   PL={pl_str}   SPM={spm_str}")


if __name__ == '__main__':
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Device: {device}")

    pl_tok = load_tokenizer('pl')
    spm_tok = load_tokenizer('spm')

    for task_name in TASKS:
        try:
            analyze_task(task_name, device, pl_tok, spm_tok)
        except Exception as e:
            print(f"\n  [{task_name}] Error: {e}")
import os
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from config import ModelConfig
from klej.tasks import TASKS
from klej.dataset import load_klej_datasets
from klej.classifier import KLEJClassifier
from klej.train_lora import load_tokenizer


def compute_metrics(task_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    task_config = TASKS[task_name]
    metric = task_config.metric

    results = {}

    if metric == 'accuracy':
        results['accuracy'] = accuracy_score(y_true, y_pred)
    elif metric == 'f1':
        results['f1'] = f1_score(y_true, y_pred, average='binary')
    elif metric == 'spearman':
        results['spearman'], _ = spearmanr(y_true, y_pred)
    elif metric == 'wmae':
        # Weighted mean absolute error (converted to score)
        # KLEJ uses: score = 1 - MAE/4 (ratings are 1-5, max error is 4)
        mae = np.mean(np.abs(y_true - y_pred))
        results['wmae'] = mae
        results['score'] = 1 - mae / 4

    return results


def evaluate_model(
    task_name: str,
    mode: str,
    checkpoint_path: str,
    split: str = 'dev',
    max_length: int = 256,
    batch_size: int = 32,
):
    """
    Evaluate a fine-tuned model on dev or generate test predictions.

    Args:
        task_name: KLEJ task name
        mode: Tokenizer mode ('pl' or 'spm')
        checkpoint_path: Path to fine-tuned model checkpoint
        split: 'dev' for evaluation, 'test' for submission
        max_length: Maximum sequence length
        batch_size: Batch size for inference

    Returns:
        For dev: dict with metrics
        For test: numpy array with predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Load tokenizer
    tokenizer = load_tokenizer(mode)

    # Load dataset
    datasets = load_klej_datasets(
        task_name=task_name,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    if split == 'dev' and 'dev' not in datasets:
        print(f"Task {task_name} has no dev set, using train for validation")
        split = 'train'

    dataset = datasets[split]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = KLEJClassifier.load_from_checkpoint(
        checkpoint_path,
        task_name=task_name,
        vocab_size=ModelConfig.vocab_size,
        d_model=ModelConfig.d_model,
        num_heads=ModelConfig.num_heads,
        num_layers=ModelConfig.num_layers,
        context_size=ModelConfig.context_size,
        d_ff=ModelConfig.d_ff,
    )
    model = model.to(device)
    model.eval()

    # Run inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)

            if TASKS[task_name].task_type == 'classification':
                preds = torch.argmax(logits, dim=-1)
            else:
                preds = logits.squeeze(-1)

            all_preds.extend(preds.cpu().numpy())

            if 'labels' in batch:
                all_labels.extend(batch['labels'].numpy())

    all_preds = np.array(all_preds)

    # For dev split, compute metrics
    if split != 'test' and len(all_labels) > 0:
        all_labels = np.array(all_labels)
        metrics = compute_metrics(task_name, all_labels, all_preds)
        print(f"\n{task_name} ({mode}) - {split} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return metrics

    # For test split, decode predictions if needed
    if TASKS[task_name].task_type == 'classification':
        label_encoder = datasets['train'].label_encoder
        all_preds = label_encoder.inverse_transform(all_preds.astype(int))

    return all_preds


def generate_submission(
    mode: str,
    checkpoint_dir: str,
    output_path: str = 'submission.zip',
    max_length: int = 256,
):
    """
    Generate KLEJ benchmark submission file.

    Creates predictions for all tasks and packages them into a zip file.
    """
    import zipfile

    predictions = {}

    for task_name in TASKS.keys():
        checkpoint_path = os.path.join(checkpoint_dir, mode, task_name)

        # Find best checkpoint
        checkpoints = list(Path(checkpoint_path).glob('best-*.ckpt'))
        if not checkpoints:
            print(f"No checkpoint found for {task_name}, skipping")
            continue

        best_ckpt = str(checkpoints[0])
        print(f"\nGenerating predictions for {task_name}...")

        preds = evaluate_model(
            task_name=task_name,
            mode=mode,
            checkpoint_path=best_ckpt,
            split='test',
            max_length=max_length,
        )
        predictions[task_name] = preds

    # Create submission files
    submission_dir = Path('submission_tmp')
    submission_dir.mkdir(exist_ok=True)

    task_to_filename = {
        'nkjp-ner': 'nkjp-ner.tsv',
        'cdsc-e': 'cdsc-e.tsv',
        'cdsc-r': 'cdsc-r.tsv',
        'cbd': 'cbd.tsv',
        'polemo-in': 'polemo2.0-in.tsv',
        'polemo-out': 'polemo2.0-out.tsv',
        'dyk': 'dyk.tsv',
        'psc': 'psc.tsv',
        'ar': 'ar.tsv',
    }

    for task_name, preds in predictions.items():
        filename = task_to_filename.get(task_name, f'{task_name}.tsv')
        filepath = submission_dir / filename

        # Write predictions
        with open(filepath, 'w') as f:
            for pred in preds:
                f.write(f'{pred}\n')

    # Create zip file
    with zipfile.ZipFile(output_path, 'w') as zf:
        for filepath in submission_dir.iterdir():
            zf.write(filepath, filepath.name)

    # Cleanup
    import shutil
    shutil.rmtree(submission_dir)

    print(f"\nSubmission saved to {output_path}")


def compare_models(checkpoint_dir: str, max_length: int = 256):
    """Compare PL vs SPM tokenizer performance across all tasks."""
    results = {'pl': {}, 'spm': {}}

    for mode in ['pl', 'spm']:
        print(f"\n{'='*60}")
        print(f"Evaluating {mode.upper()} tokenizer models")
        print(f"{'='*60}")

        for task_name in TASKS.keys():
            checkpoint_path = os.path.join(checkpoint_dir, mode, task_name)
            checkpoints = list(Path(checkpoint_path).glob('best-*.ckpt'))

            if not checkpoints:
                print(f"  {task_name}: No checkpoint found")
                continue

            best_ckpt = str(checkpoints[0])

            # Check if task has dev set
            if TASKS[task_name].has_dev:
                metrics = evaluate_model(
                    task_name=task_name,
                    mode=mode,
                    checkpoint_path=best_ckpt,
                    split='dev',
                    max_length=max_length,
                )
                results[mode][task_name] = metrics

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON: PL vs SPM")
    print(f"{'='*60}")
    print(f"{'Task':<15} {'Metric':<12} {'PL':<10} {'SPM':<10} {'Winner':<8}")
    print("-" * 55)

    pl_scores = []
    spm_scores = []

    for task_name in TASKS.keys():
        if task_name in results['pl'] and task_name in results['spm']:
            metric = TASKS[task_name].metric

            # For AR, use 1-wMAE (the 'score' key) as the KLEJ leaderboard does
            if metric == 'wmae':
                pl_score = results['pl'][task_name]['score']
                spm_score = results['spm'][task_name]['score']
                display_metric = '1-wMAE'
            else:
                pl_score = list(results['pl'][task_name].values())[0]
                spm_score = list(results['spm'][task_name].values())[0]
                display_metric = metric

            winner = 'PL' if pl_score > spm_score else 'SPM'
            print(f"{task_name:<15} {display_metric:<12} {pl_score:<10.4f} {spm_score:<10.4f} {winner:<8}")

            pl_scores.append(pl_score)
            spm_scores.append(spm_score)

    if pl_scores:
        pl_avg = sum(pl_scores) / len(pl_scores)
        spm_avg = sum(spm_scores) / len(spm_scores)
        avg_winner = 'PL' if pl_avg > spm_avg else 'SPM'
        print("-" * 55)
        print(f"{'AVERAGE':<15} {'(KLEJ avg)':<12} {pl_avg:<10.4f} {spm_avg:<10.4f} {avg_winner:<8}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate KLEJ models')

    parser.add_argument('--action', type=str, required=True,
                       choices=['evaluate', 'submit', 'compare'],
                       help='Action to perform')
    parser.add_argument('--task', type=str, choices=list(TASKS.keys()),
                       help='KLEJ task name (for evaluate)')
    parser.add_argument('--mode', type=str, choices=['pl', 'spm'],
                       help='Tokenizer mode')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint (for evaluate)')
    parser.add_argument('--checkpoint_dir', type=str, default='models/klej',
                       help='Directory with all checkpoints (for submit/compare)')
    parser.add_argument('--output', type=str, default='submission.zip',
                       help='Output path for submission')
    parser.add_argument('--max_length', type=int, default=256)

    args = parser.parse_args()

    if args.action == 'evaluate':
        if not args.task or not args.mode or not args.checkpoint:
            parser.error('--task, --mode, and --checkpoint required for evaluate')
        evaluate_model(
            task_name=args.task,
            mode=args.mode,
            checkpoint_path=args.checkpoint,
            split='dev',
            max_length=args.max_length,
        )
    elif args.action == 'submit':
        if not args.mode:
            parser.error('--mode required for submit')
        generate_submission(
            mode=args.mode,
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output,
            max_length=args.max_length,
        )
    elif args.action == 'compare':
        compare_models(
            checkpoint_dir=args.checkpoint_dir,
            max_length=args.max_length,
        )


if __name__ == '__main__':
    main()

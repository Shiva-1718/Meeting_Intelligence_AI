
from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from data_loading import build_decision_dataset

MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "decision_classifier")


class TurnDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=32):
        enc = tokenizer(
            [r["text"] for r in rows],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = [r["label"] for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "labels": torch.tensor(self.labels[idx]),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    print("Building labelled dataset from the 47 decision-annotated AMI meetings...")
    splits = build_decision_dataset()
    for name, rows in splits.items():
        pos = sum(r["label"] for r in rows)
        print(f"  {name}: {len(rows)} turns, {pos} positive ({pos / max(len(rows), 1):.1%})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TurnDataset(splits["train"], tokenizer)
    val_ds = TurnDataset(splits["val"], tokenizer)
    test_ds = TurnDataset(splits["test"], tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("\nTraining (early-stops on best val F1)...")
    trainer.train()

    print("\nTest-set performance (held-out meetings):")
    test_metrics = trainer.evaluate(test_ds)
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    test_labels = [r["label"] for r in splits["test"]]
    baseline_f1 = precision_recall_fscore_support(
        test_labels, [0] * len(test_labels), average="binary", zero_division=0
    )[2]
    print(f"  majority-class baseline F1: {baseline_f1:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nSaved fine-tuned model to {OUT_DIR}")

    _plot_loss(trainer.state.log_history)


def _plot_loss(log_history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_loss = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e]
    val_loss = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

    plt.figure(figsize=(6, 4))
    if train_loss:
        plt.plot(*zip(*train_loss), marker="o", label="train loss")
    if val_loss:
        plt.plot(*zip(*val_loss), marker="o", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Decision classifier: train vs val loss")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_curve.png")
    plt.savefig(out_path)
    print(f"Saved loss curve to {out_path}")


if __name__ == "__main__":
    main()

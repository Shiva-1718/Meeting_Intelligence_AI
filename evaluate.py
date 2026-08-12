from __future__ import annotations

import os

os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

from rouge_score import rouge_scorer
from sklearn.metrics import classification_report

from data_loading import build_decision_dataset, load_qmsum
from pipeline import classify_decisions, summarize_abstractive, summarize_extractive, summarize_overall


def eval_decision_classifier():
    print("=" * 60)
    print("Decision classifier (test split = held-out AMI meetings)")
    print("=" * 60)
    test_rows = build_decision_dataset()["test"]
    turns = [{"text": r["text"], "turn_id": None, "start": None, "speaker": None} for r in test_rows]
    true_labels = [r["label"] for r in test_rows]

    predicted = classify_decisions(turns)
    predicted_texts = {p["reference"] for p in predicted}
    pred_labels = [int(r["text"] in predicted_texts) for r in test_rows]

    print(classification_report(true_labels, pred_labels, target_names=["other", "decision"], zero_division=0))

    baseline_labels = [0] * len(true_labels)
    print("Majority-class baseline (always 'other'):")
    print(classification_report(true_labels, baseline_labels, target_names=["other", "decision"], zero_division=0))


def eval_summarisation(n_meetings: int = 10):
    print("=" * 60)
    print(f"Summarisation: extractive baseline vs abstractive (QMSum test, n={n_meetings})")
    print("=" * 60)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    records = load_qmsum(domain="ALL", split="test")[:n_meetings]

    totals = {"extractive": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0},
              "abstractive": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}}

    for rec in records:
        turns = [{"text": t["content"]} for t in rec["meeting_transcripts"]]
        reference = rec["general_query_list"][0]["answer"]

        ext_summary = summarize_extractive(turns)
        abs_summary = summarize_overall(summarize_abstractive(turns))

        for name, summary in [("extractive", ext_summary), ("abstractive", abs_summary)]:
            scores = scorer.score(reference, summary)
            for metric in totals[name]:
                totals[name][metric] += scores[metric].fmeasure

    for name, metrics in totals.items():
        print(f"{name}:")
        for metric, total in metrics.items():
            print(f"  {metric}: {total / len(records):.3f}")


if __name__ == "__main__":
    eval_decision_classifier()
    print()
    eval_summarisation()

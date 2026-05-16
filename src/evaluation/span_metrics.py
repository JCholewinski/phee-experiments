# src/evaluation/span_metrics.py

from collections import defaultdict
from typing import List, Dict, Any, Tuple


def span_key(span: Dict[str, Any]) -> Tuple[str, int, int]:
    return span["label"], span["start"], span["end"]


def safe_divide(a: int, b: int) -> float:
    return a / b if b != 0 else 0.0


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_span_metrics(
    gold_spans: List[Dict[str, Any]],
    pred_spans: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Exact span-level evaluation.

    A prediction is correct only if label, start and end are identical.
    """

    gold_set = {span_key(span) for span in gold_spans}
    pred_set = {span_key(span) for span in pred_spans}

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    overall = compute_prf(tp, fp, fn)

    labels = sorted(
        {span["label"] for span in gold_spans}
        | {span["label"] for span in pred_spans}
    )

    per_label = {}

    for label in labels:
        gold_label_set = {s for s in gold_set if s[0] == label}
        pred_label_set = {s for s in pred_set if s[0] == label}

        label_tp = len(gold_label_set & pred_label_set)
        label_fp = len(pred_label_set - gold_label_set)
        label_fn = len(gold_label_set - pred_label_set)

        per_label[label] = compute_prf(label_tp, label_fp, label_fn)

    return {
        "overall": overall,
        "per_label": per_label,
    }


def compute_dataset_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_gold = []
    all_pred = []

    for record in records:
        all_gold.extend(record.get("gold_spans", []))
        all_pred.extend(record.get("pred_spans", []))

    return compute_span_metrics(all_gold, all_pred)
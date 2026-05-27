# src/evaluation/span_metrics.py

from collections import defaultdict
from typing import List, Dict, Any, Tuple

def safe_divide(a, b):
    return a / b if b != 0 else 0.0


def precision_recall_f1(tp, fp, fn):
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def span_key(span):
    return (
        span["label"],
        span["start"],
        span["end"],
    )


def trigger_identification_key(span):
    return (
        "TRIGGER",
        span["start"],
        span["end"],
    )


def trigger_classification_key(span):
    return (
        "TRIGGER",
        span.get("event_type"),
        span["start"],
        span["end"],
    )

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

    metrics = compute_span_metrics(all_gold, all_pred)

    trigger_metrics = compute_trigger_metrics(records)

    metrics["trigger_identification"] = trigger_metrics["trigger_identification"]
    metrics["trigger_classification"] = trigger_metrics["trigger_classification"]
    metrics["event_classification"] = trigger_metrics["event_classification"]

    return metrics

def compute_set_metrics(gold_items, pred_items):
    gold_items = set(gold_items)
    pred_items = set(pred_items)

    tp = len(gold_items & pred_items)
    fp = len(pred_items - gold_items)
    fn = len(gold_items - pred_items)

    return precision_recall_f1(tp, fp, fn)

def compute_trigger_metrics(records):
    gold_trigger_i = []
    pred_trigger_i = []

    gold_trigger_c = []
    pred_trigger_c = []

    gold_event_types = []
    pred_event_types = []

    for record in records:
        gold_spans = record.get("gold_spans", [])
        pred_spans = record.get("pred_spans", [])

        gold_triggers = [
            span for span in gold_spans
            if span["label"] == "TRIGGER"
        ]

        pred_triggers = [
            span for span in pred_spans
            if span["label"] == "TRIGGER"
        ]

        for span in gold_triggers:
            gold_trigger_i.append((
                record["id"],
                trigger_identification_key(span),
            ))
            gold_trigger_c.append((
                record["id"],
                trigger_classification_key(span),
            ))

        for span in pred_triggers:
            pred_trigger_i.append((
                record["id"],
                trigger_identification_key(span),
            ))
            pred_trigger_c.append((
                record["id"],
                trigger_classification_key(span),
            ))

        gold_types = {
            span.get("event_type")
            for span in gold_triggers
            if span.get("event_type") is not None
        }

        pred_types = {
            span.get("event_type")
            for span in pred_triggers
            if span.get("event_type") is not None
        }

        for event_type in gold_types:
            gold_event_types.append((record["id"], event_type))

        for event_type in pred_types:
            pred_event_types.append((record["id"], event_type))

    return {
        "trigger_identification": compute_set_metrics(
            gold_trigger_i,
            pred_trigger_i,
        ),
        "trigger_classification": compute_set_metrics(
            gold_trigger_c,
            pred_trigger_c,
        ),
        "event_classification": compute_set_metrics(
            gold_event_types,
            pred_event_types,
        ),
    }
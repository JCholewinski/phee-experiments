"""Multi-label event type classifier (ADE, PTE), the first stage of the extractive QA
pipeline (as in the existing implementation)."""

import torch

from phee.schema import EVENT_TYPES
from phee.utils import autocast_context, get_device


def features(samples, tokenizer, max_length=256, with_labels=True):
    encoded = tokenizer([sample["context"] for sample in samples], truncation=True, max_length=max_length)
    items = []
    for i, sample in enumerate(samples):
        item = {"input_ids": encoded["input_ids"][i]}
        if with_labels:
            present = {event["event_type"] for event in sample["events"]}
            item["labels"] = [1.0 if event_type in present else 0.0 for event_type in EVENT_TYPES]
        items.append(item)
    return items


def make_collate(pad_token_id):
    def collate(items):
        length = max(len(item["input_ids"]) for item in items)
        input_ids = torch.full((len(items), length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(items), length), dtype=torch.long)
        for row, item in enumerate(items):
            n = len(item["input_ids"])
            input_ids[row, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[row, :n] = 1
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "labels" in items[0]:
            batch["labels"] = torch.tensor([item["labels"] for item in items], dtype=torch.float)
        return batch

    return collate


def predict(model, tokenizer, samples, settings, threshold=0.5, force_one=True, max_length=256):
    """Predicted event types per sample. With `force_one`, the most probable type is
    kept when no type passes the threshold (every PHEE sentence has an event)."""
    device = get_device()
    items = features(samples, tokenizer, max_length=max_length, with_labels=False)
    collate = make_collate(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    predictions = []
    batch_size = settings.get("eval_batch_size", 64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = collate(items[start:start + batch_size])
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(settings.get("precision", "bf16")):
                logits = model(**batch).logits
            probabilities = torch.sigmoid(logits.float()).cpu().tolist()
            for row in probabilities:
                types = [t for t, p in zip(EVENT_TYPES, row) if p >= threshold]
                if not types and force_one:
                    types = [EVENT_TYPES[max(range(len(row)), key=lambda k: row[k])]]
                predictions.append({"types": types, "scores": dict(zip(EVENT_TYPES, row))})
    return predictions


def score(samples, predictions):
    tp = fp = fn = 0
    for sample, prediction in zip(samples, predictions):
        gold = {event["event_type"] for event in sample["events"]}
        pred = set(prediction["types"])
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"p": precision, "r": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}

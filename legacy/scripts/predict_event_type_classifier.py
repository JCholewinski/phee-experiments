# scripts/predict_event_type_classifier.py

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


EVENT_LABELS = ["ADE", "PTE"]

RAW_LABEL_TO_EVENT_TYPE = {
    "Adverse_event": "ADE",
    "Potential_therapeutic_event": "PTE",
}


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_jsonl(records, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_split_path(config, split):
    return config["data"][f"{split}_path"]


def get_gold_event_types(sample):
    event_types = set()

    for event in sample["event"]:
        for _, _, raw_label in event:
            if raw_label in RAW_LABEL_TO_EVENT_TYPE:
                event_types.add(RAW_LABEL_TO_EVENT_TYPE[raw_label])

    return sorted(event_types)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/event_type_classifier.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = get_split_path(config, args.split)
    model_path = args.model_path or config["training"]["output_dir_final"]
    threshold = args.threshold or config["prediction"].get("threshold", 0.5)

    output_path = args.output_path
    if output_path is None:
        output_path = f"outputs/predictions/event_type_{args.split}.jsonl"

    samples = load_jsonl(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_records = []
    sample_times = []

    for record_index, sample in enumerate(samples):
        start_time = time.perf_counter()

        tokens = sample["sentence"]
        text = " ".join(str(token) for token in tokens)

        inputs = tokenizer(
            text,
            truncation=True,
            max_length=config["training"].get("max_length", 256),
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        scores = torch.sigmoid(outputs.logits)[0].cpu().tolist()

        predicted_event_types = [
            label
            for label, score in zip(EVENT_LABELS, scores)
            if score >= threshold
        ]

        end_time = time.perf_counter()
        sample_times.append(end_time - start_time)

        output_records.append(
            {
                "record_index": record_index,
                "id": sample.get("id", record_index),
                "tokens": tokens,
                "text": text,
                "scores": {
                    label: float(score)
                    for label, score in zip(EVENT_LABELS, scores)
                },
                "predicted_event_types": predicted_event_types,
                "gold_event_types": get_gold_event_types(sample),
                "inference_time_seconds": end_time - start_time,
            }
        )

    save_jsonl(output_records, output_path)

    timing = {
        "method": "event_type_classifier",
        "split": args.split,
        "num_samples": len(samples),
        "total_inference_time_seconds": sum(sample_times),
        "avg_inference_time_seconds": sum(sample_times) / len(sample_times),
        "median_inference_time_seconds": statistics.median(sample_times),
        "min_inference_time_seconds": min(sample_times),
        "max_inference_time_seconds": max(sample_times),
        "model_path": model_path,
        "device": str(device),
        "threshold": threshold,
    }

    timing_output_path = Path(
        f"outputs/timing/event_type_{args.split}_inference_timing.json"
    )
    timing_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(timing_output_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    print(json.dumps(timing, indent=2, ensure_ascii=False))
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
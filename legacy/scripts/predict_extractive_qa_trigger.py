# scripts/predict_extractive_qa_trigger.py

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.preprocessing.to_extractive_qa import (
    EVENT_TYPE_TO_QUESTION,
    TRIGGER_LABEL_TO_EVENT_TYPE,
    tokens_to_context_and_offsets,
)


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


def get_raw_split_path(config, split):
    if "raw_data" in config:
        return config["raw_data"][f"{split}_path"]

    raise KeyError(
        "Missing raw_data section in config. "
        "Add raw_data.train_path / raw_data.val_path / raw_data.test_path."
    )


def token_span_to_text(tokens, start, end):
    return " ".join(str(token) for token in tokens[start : end + 1])


def extract_gold_trigger_spans(sample):
    tokens = sample["sentence"]
    context, token_offsets = tokens_to_context_and_offsets(tokens)

    gold_spans = []

    for event in sample["event"]:
        for start, end, raw_label in event:
            if raw_label not in TRIGGER_LABEL_TO_EVENT_TYPE:
                continue

            char_start = token_offsets[start][0]
            char_end = token_offsets[end][1]

            gold_spans.append(
                {
                    "label": "TRIGGER",
                    "start": start,
                    "end": end,
                    "text": context[char_start:char_end],
                    "event_type": TRIGGER_LABEL_TO_EVENT_TYPE[raw_label],
                }
            )

    return gold_spans


def char_span_to_token_span(char_start, char_end, token_offsets):
    overlapping_token_indices = []

    for token_idx, (token_start, token_end) in enumerate(token_offsets):
        if token_end <= char_start:
            continue

        if token_start >= char_end:
            continue

        overlapping_token_indices.append(token_idx)

    if not overlapping_token_indices:
        return None, None

    return overlapping_token_indices[0], overlapping_token_indices[-1]


def spans_overlap(span_a, span_b):
    return not (
        span_a["char_end"] <= span_b["char_start"]
        or span_b["char_end"] <= span_a["char_start"]
    )


def predict_n_best_answers(
    model,
    tokenizer,
    question,
    context,
    device,
    max_length=384,
    doc_stride=128,
    max_answer_length=8,
    n_best_size=50,
    max_spans=1,
):
    tokenized = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    sequence_ids_per_feature = [
        tokenized.sequence_ids(i)
        for i in range(tokenized["input_ids"].shape[0])
    ]

    offset_mapping = tokenized.pop("offset_mapping")

    model_inputs = {
        key: value.to(device)
        for key, value in tokenized.items()
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    start_logits = outputs.start_logits.cpu()
    end_logits = outputs.end_logits.cpu()

    candidates = []

    for feature_idx in range(model_inputs["input_ids"].shape[0]):
        sequence_ids = sequence_ids_per_feature[feature_idx]
        offsets = offset_mapping[feature_idx].tolist()

        start_scores = start_logits[feature_idx]
        end_scores = end_logits[feature_idx]

        context_positions = []

        for idx, sequence_id in enumerate(sequence_ids):
            if sequence_id != 1:
                continue

            char_start, char_end = offsets[idx]

            if char_start == char_end:
                continue

            context_positions.append(idx)

        if not context_positions:
            continue

        top_start_indexes = sorted(
            context_positions,
            key=lambda idx: float(start_scores[idx]),
            reverse=True,
        )[:n_best_size]

        top_end_indexes = sorted(
            context_positions,
            key=lambda idx: float(end_scores[idx]),
            reverse=True,
        )[:n_best_size]

        for start_index in top_start_indexes:
            for end_index in top_end_indexes:
                if end_index < start_index:
                    continue

                answer_length = end_index - start_index + 1

                if answer_length > max_answer_length:
                    continue

                char_start, _ = offsets[start_index]
                _, char_end = offsets[end_index]

                if char_start == char_end:
                    continue

                score = float(start_scores[start_index] + end_scores[end_index])

                candidates.append(
                    {
                        "text": context[char_start:char_end],
                        "char_start": char_start,
                        "char_end": char_end,
                        "score": score,
                    }
                )

    candidates = sorted(
        candidates,
        key=lambda candidate: candidate["score"],
        reverse=True,
    )

    selected = []

    for candidate in candidates:
        if any(spans_overlap(candidate, previous) for previous in selected):
            continue

        selected.append(candidate)

        if len(selected) >= max_spans:
            break

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/extractive_qa_trigger.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--event_predictions_path", default=None)
    parser.add_argument("--output_path", default=None)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_data_path = get_raw_split_path(config, args.split)
    model_path = args.model_path or config["training"]["output_dir_final"]

    event_predictions_path = args.event_predictions_path
    if event_predictions_path is None:
        event_predictions_path = f"outputs/predictions/event_type_{args.split}.jsonl"

    output_path = args.output_path
    if output_path is None:
        output_path = f"outputs/predictions/extractive_qa_trigger_{args.split}.jsonl"

    max_length = config["training"].get("max_length", 384)
    doc_stride = config["training"].get("doc_stride", 128)
    max_answer_length = config["training"].get("max_answer_length", 8)

    n_best_size = config.get("prediction", {}).get("n_best_size", 50)
    max_spans_per_event_type = config.get("prediction", {}).get(
        "max_spans_per_event_type",
        1,
    )

    raw_samples = load_jsonl(raw_data_path)
    event_predictions = load_jsonl(event_predictions_path)

    if len(raw_samples) != len(event_predictions):
        raise ValueError(
            f"Raw samples and event predictions length mismatch: "
            f"{len(raw_samples)} vs {len(event_predictions)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_records = []
    sample_times = []

    for record_index, (sample, event_pred) in enumerate(
        zip(raw_samples, event_predictions)
    ):
        sample_start = time.perf_counter()

        tokens = sample["sentence"]
        context, token_offsets = tokens_to_context_and_offsets(tokens)

        gold_spans = extract_gold_trigger_spans(sample)
        pred_spans = []

        predicted_event_types = event_pred.get("predicted_event_types", [])

        for event_type in predicted_event_types:
            if event_type not in EVENT_TYPE_TO_QUESTION:
                continue

            question = EVENT_TYPE_TO_QUESTION[event_type]

            answers = predict_n_best_answers(
                model=model,
                tokenizer=tokenizer,
                question=question,
                context=context,
                device=device,
                max_length=max_length,
                doc_stride=doc_stride,
                max_answer_length=max_answer_length,
                n_best_size=n_best_size,
                max_spans=max_spans_per_event_type,
            )

            for answer in answers:
                token_start, token_end = char_span_to_token_span(
                    char_start=answer["char_start"],
                    char_end=answer["char_end"],
                    token_offsets=token_offsets,
                )

                if token_start is None or token_end is None:
                    continue

                pred_spans.append(
                    {
                        "label": "TRIGGER",
                        "start": token_start,
                        "end": token_end,
                        "text": answer["text"],
                        "event_type": event_type,
                        "score": answer["score"],
                    }
                )

        sample_end = time.perf_counter()
        inference_time = sample_end - sample_start
        sample_times.append(inference_time)

        output_records.append(
            {
                "id": record_index,
                "sample_id": sample.get("id", record_index),
                "tokens": tokens,
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
                "predicted_event_types": predicted_event_types,
                "event_type_scores": event_pred.get("scores", {}),
                "inference_time_seconds": inference_time,
            }
        )

    save_jsonl(output_records, output_path)

    total_time = sum(sample_times)

    timing = {
        "method": "extractive_qa_trigger_pipeline",
        "split": args.split,
        "num_samples": len(sample_times),
        "total_inference_time_seconds": total_time,
        "avg_inference_time_seconds": total_time / len(sample_times),
        "median_inference_time_seconds": statistics.median(sample_times),
        "min_inference_time_seconds": min(sample_times),
        "max_inference_time_seconds": max(sample_times),
        "model_path": model_path,
        "event_predictions_path": event_predictions_path,
        "device": str(device),
    }

    timing_output_path = Path(
        f"outputs/timing/extractive_qa_trigger_{args.split}_inference_timing.json"
    )
    timing_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(timing_output_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    print(json.dumps(timing, indent=2, ensure_ascii=False))
    print(f"Saved predictions to: {output_path}")
    print(f"Number of output records: {len(output_records)}")


if __name__ == "__main__":
    main()
# scripts/predict_extractive_qa_trigger.py

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


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
    key = f"{split}_path"

    if key not in config["data"]:
        available = ", ".join(config["data"].keys())
        raise KeyError(
            f"Missing data.{key} in config. Available data keys: {available}"
        )

    return config["data"][key]


def char_span_to_token_span(char_start, char_end, token_offsets):
    """
    Maps predicted character span back to original token indices.
    char_end is exclusive.
    """

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


def predict_answer(
    model,
    tokenizer,
    question,
    context,
    device,
    max_length=384,
    doc_stride=128,
    max_answer_length=30,
):
    inputs = tokenizer(
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

    offset_mapping = inputs.pop("offset_mapping")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits.cpu()
    end_logits = outputs.end_logits.cpu()

    best_score = None
    best_answer = {
        "text": "",
        "char_start": None,
        "char_end": None,
        "score": None,
    }

    for feature_idx in range(input_ids.shape[0]):
        sequence_ids = inputs_to_sequence_ids(
            tokenizer=tokenizer,
            input_ids=input_ids[feature_idx],
        )

        offsets = offset_mapping[feature_idx].tolist()

        start_scores = start_logits[feature_idx]
        end_scores = end_logits[feature_idx]

        # top-k zamiast pełnej macierzy, żeby było prościej i szybciej
        start_indexes = torch.topk(start_scores, k=min(20, len(start_scores))).indices.tolist()
        end_indexes = torch.topk(end_scores, k=min(20, len(end_scores))).indices.tolist()

        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= len(offsets) or end_index >= len(offsets):
                    continue

                if sequence_ids[start_index] != 1 or sequence_ids[end_index] != 1:
                    continue

                if end_index < start_index:
                    continue

                answer_length = end_index - start_index + 1

                if answer_length > max_answer_length:
                    continue

                char_start, _ = offsets[start_index]
                _, char_end = offsets[end_index]

                if char_start is None or char_end is None:
                    continue

                if char_start == char_end:
                    continue

                score = float(start_scores[start_index] + end_scores[end_index])

                if best_score is None or score > best_score:
                    best_score = score
                    best_answer = {
                        "text": context[char_start:char_end],
                        "char_start": char_start,
                        "char_end": char_end,
                        "score": score,
                    }

    return best_answer


def inputs_to_sequence_ids(tokenizer, input_ids):
    """
    Reconstructs sequence_ids for a single encoded example.

    sequence_ids:
    None = special tokens
    0 = question
    1 = context

    Dla tokenizerów fast wygodniej byłoby użyć tokenized.sequence_ids(i),
    ale po return_tensors tracimy BatchEncoding z tą metodą per feature
    w prosty sposób, więc rekonstruujemy po token_type_ids jeśli są,
    a dla DeBERTa fallback robimy przez special tokens.
    """

    # Dla BERT-like tokenizerów często są token_type_ids.
    # Ale DeBERTa często ich nie używa, więc obsługujemy fallback.
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    sep_token = tokenizer.sep_token

    sequence_ids = []
    current_sequence = 0

    sep_seen = 0

    for token in tokens:
        if token in {
            tokenizer.cls_token,
            tokenizer.sep_token,
            tokenizer.pad_token,
        }:
            sequence_ids.append(None)

            if token == sep_token:
                sep_seen += 1
                if sep_seen == 1:
                    current_sequence = 1

            continue

        sequence_ids.append(current_sequence)

    return sequence_ids


def build_output_records(qa_examples, predictions_by_sample):
    output_records = []

    for sample_id, grouped in predictions_by_sample.items():
        tokens = grouped["tokens"]
        gold_spans = grouped["gold_spans"]
        pred_spans = grouped["pred_spans"]
        inference_times = grouped["inference_times"]

        output_records.append(
            {
                "id": sample_id,
                "tokens": tokens,
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
                "inference_time_seconds": sum(inference_times),
            }
        )

    return output_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/extractive_qa_trigger.yaml")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to trained model. If not given, uses training.output_dir_final.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Where to save JSONL predictions.",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = get_split_path(config, args.split)
    model_path = args.model_path or config["training"]["output_dir_final"]

    output_path = args.output_path
    if output_path is None:
        output_path = f"outputs/predictions/extractive_qa_trigger_{args.split}.jsonl"

    max_length = config["training"].get("max_length", 384)
    doc_stride = config["training"].get("doc_stride", 128)
    max_answer_length = config["training"].get("max_answer_length", 30)

    qa_examples = load_jsonl(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions_by_sample = {}
    sample_times = []

    for example in qa_examples:
        start_time = time.perf_counter()

        answer = predict_answer(
            model=model,
            tokenizer=tokenizer,
            question=example["question"],
            context=example["context"],
            device=device,
            max_length=max_length,
            doc_stride=doc_stride,
            max_answer_length=max_answer_length,
        )

        end_time = time.perf_counter()
        inference_time = end_time - start_time
        sample_times.append(inference_time)

        sample_id = example["sample_id"]

        if sample_id not in predictions_by_sample:
            predictions_by_sample[sample_id] = {
                "tokens": example["tokens"],
                "gold_spans": [],
                "pred_spans": [],
                "inference_times": [],
            }

        predictions_by_sample[sample_id]["gold_spans"].append(
            example["gold_trigger"]
        )

        if answer["char_start"] is not None and answer["char_end"] is not None:
            token_start, token_end = char_span_to_token_span(
                char_start=answer["char_start"],
                char_end=answer["char_end"],
                token_offsets=example["token_offsets"],
            )

            if token_start is not None and token_end is not None:
                predictions_by_sample[sample_id]["pred_spans"].append(
                    {
                        "label": "TRIGGER",
                        "start": token_start,
                        "end": token_end,
                        "text": answer["text"],
                        "event_type": example["event_type"],
                        "score": answer["score"],
                    }
                )

        predictions_by_sample[sample_id]["inference_times"].append(inference_time)

    output_records = build_output_records(
        qa_examples=qa_examples,
        predictions_by_sample=predictions_by_sample,
    )

    save_jsonl(output_records, output_path)

    total_time = sum(sample_times)

    timing = {
        "method": "extractive_qa_trigger",
        "split": args.split,
        "num_qa_examples": len(sample_times),
        "num_output_records": len(output_records),
        "total_inference_time_seconds": total_time,
        "avg_inference_time_seconds": total_time / len(sample_times),
        "median_inference_time_seconds": statistics.median(sample_times),
        "min_inference_time_seconds": min(sample_times),
        "max_inference_time_seconds": max(sample_times),
        "model_path": model_path,
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
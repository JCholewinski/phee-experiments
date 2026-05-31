# scripts/predict_seq.py

import argparse
import json
from pathlib import Path
import time
import statistics

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.evaluation.bio_to_spans import bio_to_spans
from src.preprocessing.to_bio import convert_sample_to_bio
from src.preprocessing.tokenize_and_align import tokenize_and_align
from src.preprocessing.tokenize_and_align_crf import tokenize_and_align_crf ###ADDED###
from src.utils.labeling import ID2LABEL, LABEL2ID
from transformers import AutoTokenizer, AutoModelForTokenClassification ###ADDED###
from src.models import BertMLPForTokenClassification, BertCRFForTokenClassification ###ADDED###


def load_jsonl(path: str):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_jsonl(records, path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_split_path(config, split: str) -> str:
    key = f"{split}_path"

    if key not in config["data"]:
        available = ", ".join(config["data"].keys())
        raise KeyError(
            f"Missing data.{key} in config. Available data keys: {available}"
        )

    return config["data"][key]


def align_predictions_to_words(pred_ids, word_ids, num_words):
    """
    Convert tokenizer/subtoken-level predictions back to original word-level labels.

    For each original word/token, we take the prediction from the first tokenizer
    token that belongs to this word. This guarantees exactly one prediction per
    original token.
    """

    word_predictions = [None] * num_words

    for pred_id, word_id in zip(pred_ids, word_ids):
        if word_id is None:
            continue

        if word_id < 0 or word_id >= num_words:
            continue

        if word_predictions[word_id] is None:
            word_predictions[word_id] = ID2LABEL[int(pred_id)]

    missing = [i for i, label in enumerate(word_predictions) if label is None]

    if missing:
        raise ValueError(
            f"Could not align predictions for word indices: {missing}. "
            f"num_words={num_words}, word_ids={word_ids}"
        )

    return word_predictions


def predict_sample(model, tokenizer, processed_sample, device, head_type):
    #tokenized = tokenize_and_align(processed_sample, tokenizer, LABEL2ID)
    if head_type == "crf":
        tokenized = tokenize_and_align_crf(processed_sample, tokenizer, LABEL2ID)
    else:
        tokenized = tokenize_and_align(processed_sample, tokenizer, LABEL2ID)

    input_ids = torch.tensor(tokenized["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(tokenized["attention_mask"]).unsqueeze(0).to(device)

    if head_type == "crf":
        crf_mask = torch.tensor(tokenized["crf_mask"]).unsqueeze(0).to(device)

    # with torch.no_grad():
    #     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    #     pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    with torch.no_grad():
        if head_type == "crf":
            decoded = model.decode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                crf_mask=crf_mask,
            )

            # decoded[0] only contains labels for active CRF positions.
            # Expand it back to full tokenizer length.
            pred_ids = [LABEL2ID["O"]] * len(tokenized["input_ids"])
            active_positions = [
                i for i, m in enumerate(tokenized["crf_mask"]) if m == 1
            ]

            for pos, label_id in zip(active_positions, decoded[0]):
                pred_ids[pos] = label_id

        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

    word_ids = tokenized.word_ids()

    pred_labels = align_predictions_to_words(
        pred_ids=pred_ids,
        word_ids=word_ids,
        num_words=len(processed_sample["tokens"]),
    )

    return pred_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seq.yaml")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to run prediction on.",
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
    
    head_type = config["model"].get("head_type", "linear") ###ADDED###

    data_path = get_split_path(config, args.split)
    model_path = args.model_path or config["training"]["output_dir_final"]

    output_path = args.output_path
    if output_path is None:
        output_path = f"outputs/predictions/seq_{head_type}_{args.split}.jsonl" ###ADDED###

    raw_data = load_jsonl(data_path)
    processed_data = [convert_sample_to_bio(sample) for sample in raw_data]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #model = AutoModelForTokenClassification.from_pretrained(model_path)
    if head_type == "linear": ###ADDED###
        model = AutoModelForTokenClassification.from_pretrained(model_path)

    elif head_type == "mlp":
        model = BertMLPForTokenClassification.from_pretrained(model_path)
    
    elif head_type == "crf": ###ADDED###
        model = BertCRFForTokenClassification.from_pretrained(model_path)

    else:
        raise ValueError(f"Unsupported head_type: {head_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_records = []
    sample_times = []

    for idx, processed_sample in enumerate(processed_data):
        
        sample_start = time.perf_counter()
        
        tokens = processed_sample["tokens"]
        gold_labels = processed_sample["labels"]

        pred_labels = predict_sample(
            model=model,
            tokenizer=tokenizer,
            processed_sample=processed_sample,
            device=device,
            head_type=head_type,
        )

        if len(pred_labels) != len(tokens):
            raise ValueError(
                f"Prediction length mismatch for sample {idx}: "
                f"{len(pred_labels)} predictions vs {len(tokens)} tokens"
            )

        gold_spans = bio_to_spans(tokens, gold_labels)
        pred_spans = bio_to_spans(tokens, pred_labels)

        sample_end = time.perf_counter()
        sample_times.append(sample_end - sample_start)

        output_records.append(
            {
                "id": idx,
                "tokens": tokens,
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
                "inference_time_seconds": sample_end - sample_start,
            }
        )

    total_time = sum(sample_times)

    timing = {
        "method": f"sequence_labeling_{head_type}", ###ADDED###
        "head_type": head_type, ###ADDED###
        "split": args.split,
        "num_samples": len(sample_times),
        "total_inference_time_seconds": total_time,
        "avg_inference_time_seconds": total_time / len(sample_times),
        "median_inference_time_seconds": statistics.median(sample_times),
        "min_inference_time_seconds": min(sample_times),
        "max_inference_time_seconds": max(sample_times),
        "model_path": model_path,
        "device": str(device),
    }

    timing_output_path = Path(f"outputs/timing/seq_{head_type}_{args.split}_inference_timing.json") ###ADDED###
    timing_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(timing_output_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    print(json.dumps(timing, indent=2, ensure_ascii=False))

    save_jsonl(output_records, output_path)

    print(f"Saved predictions to: {output_path}")
    print(f"Number of samples: {len(output_records)}")


if __name__ == "__main__":
    main()
# scripts/predict_seq.py

import argparse
import json
from pathlib import Path
import time
import statistics

import torch
import yaml
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.evaluation.bio_to_spans import bio_to_spans
from src.preprocessing.to_bio import convert_sample_to_bio
from src.preprocessing.tokenize_and_align import tokenize_and_align
from src.preprocessing.tokenize_and_align_crf import tokenize_and_align_crf
from src.utils.labeling import ID2LABEL, LABEL2ID

from src.models import (
    BertMLPForTokenClassification,
    BertCRFForTokenClassification,
    BertFrozenLinearCRFForTokenClassification,
)


CRF_HEAD_TYPES = {"crf", "linear_crf_frozen"}


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

    For each original word/token, we take the prediction from the first tokenizer token
    that belongs to this word. This guarantees exactly one prediction per original token.
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


def expand_crf_decoded_to_token_level(decoded_ids, tokenized):
    """
    CRF decode returns labels only for compact active positions selected by crf_mask.

    This function expands them back to tokenizer-level predictions so that the existing
    align_predictions_to_words(...) function can map predictions to original words.

    Example:
    input tokenizer positions:
    [CLS], word1, ##sub, word2, [SEP], [PAD]
    crf_mask:
       1      1      0     1     0      0
    decoded_ids:
    [O, label_word1, label_word2]

    expanded:
    [O, label_word1, O, label_word2, O, O]
    """
    crf_mask = tokenized["crf_mask"]

    if sum(crf_mask) != len(decoded_ids):
        raise ValueError(
            f"CRF decoded length mismatch: decoded={len(decoded_ids)}, "
            f"active_positions={sum(crf_mask)}"
        )

    expanded = []
    decoded_idx = 0

    for is_active in crf_mask:
        if is_active:
            expanded.append(decoded_ids[decoded_idx])
            decoded_idx += 1
        else:
            expanded.append(LABEL2ID["O"])

    return expanded


def predict_sample(model, tokenizer, processed_sample, device, head_type: str):
    if head_type in CRF_HEAD_TYPES:
        tokenized = tokenize_and_align_crf(processed_sample, tokenizer, LABEL2ID)
    else:
        tokenized = tokenize_and_align(processed_sample, tokenizer, LABEL2ID)

    input_ids = torch.tensor(tokenized["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(tokenized["attention_mask"]).unsqueeze(0).to(device)

    crf_mask = None
    if head_type in CRF_HEAD_TYPES:
        crf_mask = torch.tensor(tokenized["crf_mask"]).unsqueeze(0).to(device)

    with torch.no_grad():
        if head_type in CRF_HEAD_TYPES:
            decoded = model.decode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                crf_mask=crf_mask,
            )

            # decoded[0] contains predictions only for active CRF positions
            pred_ids = expand_crf_decoded_to_token_level(
                decoded_ids=decoded[0],
                tokenized=tokenized,
            )

        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

    word_ids = tokenized.word_ids()

    pred_labels = align_predictions_to_words(
        pred_ids=pred_ids,
        word_ids=word_ids,
        num_words=len(processed_sample["tokens"]),
    )

    return pred_labels


def load_model(head_type: str, model_path: str):
    if head_type == "linear":
        return AutoModelForTokenClassification.from_pretrained(model_path)

    if head_type == "mlp":
        return BertMLPForTokenClassification.from_pretrained(model_path)

    if head_type == "crf":
        return BertCRFForTokenClassification.from_pretrained(model_path)

    if head_type == "linear_crf_frozen":
        return BertFrozenLinearCRFForTokenClassification.from_pretrained(model_path)

    raise ValueError(f"Unsupported head_type: {head_type}")


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

    head_type = config["model"].get("head_type", "linear")

    data_path = get_split_path(config, args.split)
    model_path = args.model_path or config["training"]["output_dir_final"]

    if args.output_path is None:
        output_path = f"data/outputs/predictions/seq_{head_type}_{args.split}.jsonl"
    else:
        output_path = args.output_path

    raw_data = load_jsonl(data_path)
    processed_data = [convert_sample_to_bio(sample) for sample in raw_data]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_model(head_type=head_type, model_path=model_path)

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
        "method": f"sequence_labeling_{head_type}",
        "head_type": head_type,
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

    timing_output_path = Path(
        f"data/outputs/timing/seq_{head_type}_{args.split}_inference_timing.json"
    )
    timing_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(timing_output_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    print(json.dumps(timing, indent=2, ensure_ascii=False))

    save_jsonl(output_records, output_path)

    print(f"Saved predictions to: {output_path}")
    print(f"Number of samples: {len(output_records)}")


if __name__ == "__main__":
    main()
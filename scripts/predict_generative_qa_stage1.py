import argparse
import json
import statistics
import time
from pathlib import Path
from tqdm.auto import tqdm

import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.preprocessing.to_generative_qa import (
    build_stage1_input,
    extract_gold_stage1_spans,
    load_jsonl,
    save_jsonl,
    stage1_output_to_pred_spans,
)


def get_raw_split_path(config, split):
    return config["raw_data"][f"{split}_path"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generative_qa_stage1.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = get_raw_split_path(config, args.split)
    model_path = args.model_path or config["training"]["output_dir_final"]

    output_path = (
        args.output_path
        or f"outputs/predictions/generative_qa_stage1_{args.split}.jsonl"
    )

    max_source_length = config["training"].get("max_source_length", 512)
    max_new_tokens = config["prediction"].get("max_new_tokens", 256)
    num_beams = config["prediction"].get("num_beams", 4)

    samples = load_jsonl(raw_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}", flush=True)
    print(f"Number of samples: {len(samples)}", flush=True)
    print(f"max_new_tokens: {max_new_tokens}, num_beams: {num_beams}", flush=True)

    output_records = []
    sample_times = []

    for idx, sample in enumerate(tqdm(samples, desc=f"Predicting stage1 {args.split}")):        
        start = time.perf_counter()

        input_text = build_stage1_input(sample)
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_source_length,
            truncation=True,
        ).to(device)

        print(f"[{idx + 1}/{len(samples)}] Starting generation...", flush=True)
        generation_start = time.perf_counter()

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

        generation_end = time.perf_counter()
        print(
            f"[{idx + 1}/{len(samples)}] Generation finished in "
            f"{generation_end - generation_start:.2f}s",
            flush=True,
        )

        generated_text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        tokens = sample["sentence"]
        gold_spans = extract_gold_stage1_spans(sample)
        pred_spans = stage1_output_to_pred_spans(generated_text, tokens)

        end = time.perf_counter()
        sample_times.append(end - start)

        output_records.append({
            "id": idx,
            "sample_id": sample.get("id", idx),
            "tokens": tokens,
            "input_text": input_text,
            "generated_text": generated_text,
            "gold_spans": gold_spans,
            "pred_spans": pred_spans,
            "inference_time_seconds": end - start,
        })

    save_jsonl(output_records, output_path)

    timing = {
        "method": "generative_qa_stage1",
        "split": args.split,
        "num_samples": len(sample_times),
        "total_inference_time_seconds": sum(sample_times),
        "avg_inference_time_seconds": sum(sample_times) / len(sample_times),
        "median_inference_time_seconds": statistics.median(sample_times),
        "min_inference_time_seconds": min(sample_times),
        "max_inference_time_seconds": max(sample_times),
        "model_path": model_path,
        "device": str(device),
    }

    timing_path = Path(
        f"outputs/timing/generative_qa_stage1_{args.split}_inference_timing.json"
    )
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    print(json.dumps(timing, indent=2, ensure_ascii=False))
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()

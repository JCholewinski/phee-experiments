import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.preprocessing.to_generative_qa import (
    SUBARGUMENT_LABELS_BY_MAIN_ARGUMENT,
    SUBARGUMENT_QUESTIONS,
    build_stage2_input,
    extract_gold_subargument_spans,
    load_jsonl,
    save_jsonl,
    stage2_answer_to_pred_spans,
    token_span_text,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generative_qa_stage2.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--stage1_predictions_path", default=None)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = config["raw_data"][f"{args.split}_path"]
    stage1_predictions_path = (
        args.stage1_predictions_path
        or config["stage1_predictions"][f"{args.split}_path"]
    )
    model_path = args.model_path or config["training"]["output_dir_final"]

    output_path = (
        args.output_path
        or f"outputs/predictions/generative_qa_stage2_{args.split}.jsonl"
    )

    max_source_length = config["training"].get("max_source_length", 512)
    max_new_tokens = config["prediction"].get("max_new_tokens", 128)
    num_beams = config["prediction"].get("num_beams", 4)

    raw_samples = load_jsonl(raw_path)
    stage1_predictions = load_jsonl(stage1_predictions_path)

    if len(raw_samples) != len(stage1_predictions):
        raise ValueError(
            f"Length mismatch: raw={len(raw_samples)}, "
            f"stage1_predictions={len(stage1_predictions)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_records = []
    sample_times = []

    for idx, (sample, stage1_record) in enumerate(zip(raw_samples, stage1_predictions)):
        sample_start = time.perf_counter()

        tokens = sample["sentence"]
        gold_spans = extract_gold_subargument_spans(sample)
        pred_spans = []

        predicted_main_arguments = [
            span for span in stage1_record.get("pred_spans", [])
            if span.get("label") in SUBARGUMENT_LABELS_BY_MAIN_ARGUMENT
        ]

        generated_answers = []

        for pred_main in predicted_main_arguments:
            main_label = pred_main["label"]
            main_text = pred_main.get("text") or token_span_text(
                tokens,
                pred_main["start"],
                pred_main["end"],
            )

            for sub_label in SUBARGUMENT_LABELS_BY_MAIN_ARGUMENT[main_label]:
                question = SUBARGUMENT_QUESTIONS[sub_label].format(
                    main_argument_text=main_text
                )
                input_text = build_stage2_input(question, sample)

                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=max_source_length,
                    truncation=True,
                ).to(device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )

                generated_answer = tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                )

                generated_answers.append({
                    "subargument_label": sub_label,
                    "source_main_argument": pred_main,
                    "question": question,
                    "generated_answer": generated_answer,
                })

                pred_spans.extend(
                    stage2_answer_to_pred_spans(
                        generated_answer=generated_answer,
                        tokens=tokens,
                        subargument_label=sub_label,
                        source_main_argument=pred_main,
                    )
                )

        sample_end = time.perf_counter()
        sample_times.append(sample_end - sample_start)

        output_records.append({
            "id": idx,
            "sample_id": sample.get("id", idx),
            "tokens": tokens,
            "gold_spans": gold_spans,
            "pred_spans": pred_spans,
            "predicted_main_arguments": predicted_main_arguments,
            "generated_answers": generated_answers,
            "inference_time_seconds": sample_end - sample_start,
        })

    save_jsonl(output_records, output_path)

    timing = {
        "method": "generative_qa_stage2",
        "split": args.split,
        "num_samples": len(sample_times),
        "total_inference_time_seconds": sum(sample_times),
        "avg_inference_time_seconds": sum(sample_times) / len(sample_times),
        "median_inference_time_seconds": statistics.median(sample_times),
        "min_inference_time_seconds": min(sample_times),
        "max_inference_time_seconds": max(sample_times),
        "model_path": model_path,
        "stage1_predictions_path": stage1_predictions_path,
        "device": str(device),
    }

    timing_path = Path(
        f"outputs/timing/generative_qa_stage2_{args.split}_inference_timing.json"
    )
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    print(json.dumps(timing, indent=2, ensure_ascii=False))
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()

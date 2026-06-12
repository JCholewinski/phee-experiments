# scripts/build_extractive_qa_main_argument_data.py

import argparse
import json
from pathlib import Path

from src.preprocessing.to_extractive_qa import convert_sample_to_main_argument_qa


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


def build_split(raw_path, trigger_predictions_path, output_path):
    raw_samples = load_jsonl(raw_path)
    trigger_predictions = load_jsonl(trigger_predictions_path)

    if len(raw_samples) != len(trigger_predictions):
        raise ValueError(
            f"Length mismatch: raw={len(raw_samples)}, "
            f"trigger_predictions={len(trigger_predictions)}"
        )

    qa_examples = []

    for record_index, (sample, trigger_record) in enumerate(
        zip(raw_samples, trigger_predictions)
    ):
        predicted_triggers = trigger_record.get("pred_spans", [])

        qa_examples.extend(
            convert_sample_to_main_argument_qa(
                sample=sample,
                predicted_triggers=predicted_triggers,
                record_index=record_index,
            )
        )

    save_jsonl(qa_examples, output_path)

    no_answer = sum(
        1
        for example in qa_examples
        if not example["answers"]["answer_start"]
    )

    print(f"Raw samples: {len(raw_samples)}")
    print(f"Trigger prediction records: {len(trigger_predictions)}")
    print(f"Main QA examples: {len(qa_examples)}")
    print(f"No-answer examples: {no_answer}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_raw_path", default="data/raw/train_main.json")
    parser.add_argument("--val_raw_path", default="data/raw/val_main.json")
    parser.add_argument("--test_raw_path", default="data/raw/test_main.json")

    parser.add_argument(
        "--train_trigger_predictions_path",
        default="outputs/predictions/extractive_qa_trigger_train.jsonl",
    )
    parser.add_argument(
        "--val_trigger_predictions_path",
        default="outputs/predictions/extractive_qa_trigger_val.jsonl",
    )
    parser.add_argument(
        "--test_trigger_predictions_path",
        default="outputs/predictions/extractive_qa_trigger_test.jsonl",
    )

    parser.add_argument(
        "--output_dir",
        default="data/processed/extractive_qa_main_arguments",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_split(
        raw_path=args.train_raw_path,
        trigger_predictions_path=args.train_trigger_predictions_path,
        output_path=output_dir / "train.jsonl",
    )

    build_split(
        raw_path=args.val_raw_path,
        trigger_predictions_path=args.val_trigger_predictions_path,
        output_path=output_dir / "val.jsonl",
    )

    build_split(
        raw_path=args.test_raw_path,
        trigger_predictions_path=args.test_trigger_predictions_path,
        output_path=output_dir / "test.jsonl",
    )


if __name__ == "__main__":
    main()
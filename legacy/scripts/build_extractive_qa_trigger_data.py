# scripts/build_extractive_qa_trigger_data.py

import argparse
import json
from pathlib import Path

from src.preprocessing.to_extractive_qa import convert_sample_to_trigger_qa


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


def build_split(input_path, output_path):
    raw_samples = load_jsonl(input_path)

    qa_examples = []

    for sample in raw_samples:
        qa_examples.extend(convert_sample_to_trigger_qa(sample))

    save_jsonl(qa_examples, output_path)

    print(f"Input samples: {len(raw_samples)}")
    print(f"QA trigger examples: {len(qa_examples)}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/raw/train_main.json")
    parser.add_argument("--val_path", default="data/raw/val_main.json")
    parser.add_argument("--test_path", default="data/raw/test_main.json")
    parser.add_argument("--output_dir", default="data/processed/extractive_qa_trigger")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_split(
        input_path=args.train_path,
        output_path=output_dir / "train.jsonl",
    )

    build_split(
        input_path=args.val_path,
        output_path=output_dir / "val.jsonl",
    )

    build_split(
        input_path=args.test_path,
        output_path=output_dir / "test.jsonl",
    )


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path

from src.preprocessing.to_generative_qa import (
    build_stage1_input,
    load_jsonl,
    save_jsonl,
    serialize_stage1_target,
)


def build_split(raw_path, output_path):
    samples = load_jsonl(raw_path)
    records = []

    for idx, sample in enumerate(samples):
        records.append({
            "id": sample.get("id", idx),
            "record_index": idx,
            "input_text": build_stage1_input(sample),
            "target_text": serialize_stage1_target(sample),
        })

    save_jsonl(records, output_path)

    print("=" * 80)
    print(f"Saved to: {output_path}")
    print(f"Examples: {len(records)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_raw_path", default="data/raw/train_main.json")
    parser.add_argument("--val_raw_path", default="data/raw/val_main.json")
    parser.add_argument("--test_raw_path", default="data/raw/test_main.json")
    parser.add_argument("--output_dir", default="data/processed/generative_qa_stage1")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_split(args.train_raw_path, output_dir / "train.jsonl")
    build_split(args.val_raw_path, output_dir / "val.jsonl")
    build_split(args.test_raw_path, output_dir / "test.jsonl")


if __name__ == "__main__":
    main()

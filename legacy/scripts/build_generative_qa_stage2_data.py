import argparse
from pathlib import Path

import yaml

from src.preprocessing.to_generative_qa import (
    convert_sample_to_stage2_examples,
    load_jsonl,
    save_jsonl,
)


def build_split(raw_path, stage1_predictions_path, output_path):
    raw_samples = load_jsonl(raw_path)
    stage1_predictions = load_jsonl(stage1_predictions_path)

    if len(raw_samples) != len(stage1_predictions):
        raise ValueError(
            f"Length mismatch: raw={len(raw_samples)}, "
            f"stage1_predictions={len(stage1_predictions)}"
        )

    examples = []

    for idx, (sample, stage1_record) in enumerate(zip(raw_samples, stage1_predictions)):
        examples.extend(
            convert_sample_to_stage2_examples(
                sample=sample,
                stage1_prediction_record=stage1_record,
                record_index=idx,
            )
        )

    save_jsonl(examples, output_path)

    no_answer = sum(1 for ex in examples if ex["target_text"] == "none")
    positive = len(examples) - no_answer

    print("=" * 80)
    print(f"Saved to: {output_path}")
    print(f"Raw samples: {len(raw_samples)}")
    print(f"Stage1 prediction records: {len(stage1_predictions)}")
    print(f"Stage2 examples: {len(examples)}")
    print(f"Positive examples: {positive}")
    print(f"No-answer examples: {no_answer}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generative_qa_stage2.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path("data/processed/generative_qa_stage2")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        build_split(
            raw_path=config["raw_data"][f"{split}_path"],
            stage1_predictions_path=config["stage1_predictions"][f"{split}_path"],
            output_path=output_dir / f"{split}.jsonl",
        )


if __name__ == "__main__":
    main()

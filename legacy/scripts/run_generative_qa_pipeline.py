import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable


def run(command):
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(command))
    print("=" * 100)

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    subprocess.run(command, check=True, env=env)


def remove_path(path):
    path = Path(path)

    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_outputs():
    print("Cleaning old Generative QA outputs...")

    dirs_to_remove = [
        "outputs/models/generative_qa_stage1",
        "outputs/models/generative_qa_stage2",
        "data/processed/generative_qa_stage1",
        "data/processed/generative_qa_stage2",
    ]

    patterns_to_remove = [
        "outputs/predictions/generative_qa_stage1_*.jsonl",
        "outputs/predictions/generative_qa_stage2_*.jsonl",
        "outputs/metrics/generative_qa_stage1_*_metrics.json",
        "outputs/metrics/generative_qa_stage2_*_metrics.json",
        "outputs/timing/generative_qa_stage1_*.json",
        "outputs/timing/generative_qa_stage2_*.json",
    ]

    for path in dirs_to_remove:
        remove_path(path)

    for pattern in patterns_to_remove:
        for path in glob.glob(pattern):
            remove_path(path)


def print_metrics(path, title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    metrics_path = Path(path)

    if not metrics_path.exists():
        print(f"Missing metrics file: {path}")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/timing").mkdir(parents=True, exist_ok=True)

    if args.clean:
        clean_outputs()

    run([
        PYTHON,
        "scripts/build_generative_qa_stage1_data.py",
    ])

    if not args.skip_training:
        run([
            PYTHON,
            "scripts/train_generative_qa.py",
            "--config",
            "configs/generative_qa_stage1.yaml",
            "--method_name",
            "generative_qa_stage1",
        ])

    for split in ["train", "val", "test"]:
        run([
            PYTHON,
            "scripts/predict_generative_qa_stage1.py",
            "--config",
            "configs/generative_qa_stage1.yaml",
            "--split",
            split,
        ])

    for split in ["val", "test"]:
        run([
            PYTHON,
            "scripts/evaluate_predictions.py",
            "--predictions_path",
            f"outputs/predictions/generative_qa_stage1_{split}.jsonl",
            "--output_path",
            f"outputs/metrics/generative_qa_stage1_{split}_metrics.json",
        ])

    run([
        PYTHON,
        "scripts/build_generative_qa_stage2_data.py",
        "--config",
        "configs/generative_qa_stage2.yaml",
    ])

    if not args.skip_training:
        run([
            PYTHON,
            "scripts/train_generative_qa.py",
            "--config",
            "configs/generative_qa_stage2.yaml",
            "--method_name",
            "generative_qa_stage2",
        ])

    for split in ["val", "test"]:
        run([
            PYTHON,
            "scripts/predict_generative_qa_stage2.py",
            "--config",
            "configs/generative_qa_stage2.yaml",
            "--split",
            split,
        ])

    for split in ["val", "test"]:
        run([
            PYTHON,
            "scripts/evaluate_predictions.py",
            "--predictions_path",
            f"outputs/predictions/generative_qa_stage2_{split}.jsonl",
            "--output_path",
            f"outputs/metrics/generative_qa_stage2_{split}_metrics.json",
        ])

    print_metrics(
        "outputs/metrics/generative_qa_stage1_test_metrics.json",
        "Generative QA Stage 1 - test metrics",
    )

    print_metrics(
        "outputs/metrics/generative_qa_stage2_test_metrics.json",
        "Generative QA Stage 2 - test metrics",
    )

    print("\nGenerative QA pipeline finished successfully.")


if __name__ == "__main__":
    main()

# scripts/run_extractive_qa_pipeline.py

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
    print("Cleaning old Extractive QA outputs...")

    dirs_to_remove = [
        "outputs/models/event_type_classifier",
        "outputs/models/extractive_qa_trigger",
        "outputs/models/extractive_qa_main_arguments",
        "outputs/models/extractive_qa_subarguments",
    ]

    patterns_to_remove = [
        "outputs/predictions/event_type_*.jsonl",
        "outputs/predictions/extractive_qa_trigger_*.jsonl",
        "outputs/predictions/extractive_qa_main_arguments_*.jsonl",
        "outputs/predictions/extractive_qa_subarguments_*.jsonl",
        "outputs/metrics/extractive_qa_trigger_*_metrics.json",
        "outputs/metrics/extractive_qa_main_arguments_*_metrics.json",
        "outputs/metrics/extractive_qa_subarguments_*_metrics.json",
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

    # ------------------------------------------------------------------
    # 1. Trigger QA data
    # ------------------------------------------------------------------

    run([
        PYTHON,
        "scripts/build_extractive_qa_trigger_data.py",
    ])

    # ------------------------------------------------------------------
    # 2. Event type classifier
    # ------------------------------------------------------------------

    if not args.skip_training:
        run([
            PYTHON,
            "scripts/train_event_type_classifier.py",
            "--config",
            "configs/event_type_classifier.yaml",
        ])

    for split in ["train", "val", "test"]:
        run([
            PYTHON,
            "scripts/predict_event_type_classifier.py",
            "--config",
            "configs/event_type_classifier.yaml",
            "--split",
            split,
        ])

    # ------------------------------------------------------------------
    # 3. Trigger QA
    # ------------------------------------------------------------------

    if not args.skip_training:
        run([
            PYTHON,
            "scripts/train_extractive_qa_trigger.py",
            "--config",
            "configs/extractive_qa_trigger.yaml",
        ])

    for split in ["train", "val", "test"]:
        run([
            PYTHON,
            "scripts/predict_extractive_qa_trigger.py",
            "--config",
            "configs/extractive_qa_trigger.yaml",
            "--split",
            split,
            "--event_predictions_path",
            f"outputs/predictions/event_type_{split}.jsonl",
        ])

    for split in ["val", "test"]:
        run([
            PYTHON,
            "scripts/evaluate_predictions.py",
            "--predictions_path",
            f"outputs/predictions/extractive_qa_trigger_{split}.jsonl",
            "--output_path",
            f"outputs/metrics/extractive_qa_trigger_{split}_metrics.json",
        ])

    # ------------------------------------------------------------------
    # 4. Main argument QA
    # ------------------------------------------------------------------

    run([
        PYTHON,
        "scripts/build_extractive_qa_main_argument_data.py",
    ])

    if not args.skip_training:
        run([
            PYTHON,
            "scripts/train_extractive_qa_main_arguments.py",
            "--config",
            "configs/extractive_qa_main_arguments.yaml",
        ])

    for split in ["train", "val", "test"]:
        run([
            PYTHON,
            "scripts/predict_extractive_qa_main_arguments.py",
            "--config",
            "configs/extractive_qa_main_arguments.yaml",
            "--split",
            split,
        ])

    for split in ["val", "test"]:
        run([
            PYTHON,
            "scripts/evaluate_predictions.py",
            "--predictions_path",
            f"outputs/predictions/extractive_qa_main_arguments_{split}.jsonl",
            "--output_path",
            f"outputs/metrics/extractive_qa_main_arguments_{split}_metrics.json",
        ])

    # ------------------------------------------------------------------
    # 5. Subargument QA
    # ------------------------------------------------------------------

    run([
        PYTHON,
        "scripts/build_extractive_qa_subargument_data.py",
    ])

    if not args.skip_training:
        run([
            PYTHON,
            "scripts/train_extractive_qa_subarguments.py",
            "--config",
            "configs/extractive_qa_subarguments.yaml",
        ])

    for split in ["val", "test"]:
        run([
            PYTHON,
            "scripts/predict_extractive_qa_subarguments.py",
            "--config",
            "configs/extractive_qa_subarguments.yaml",
            "--split",
            split,
        ])

    for split in ["val", "test"]:
        run([
            PYTHON,
            "scripts/evaluate_predictions.py",
            "--predictions_path",
            f"outputs/predictions/extractive_qa_subarguments_{split}.jsonl",
            "--output_path",
            f"outputs/metrics/extractive_qa_subarguments_{split}_metrics.json",
        ])

    # ------------------------------------------------------------------
    # 6. Print final test metrics
    # ------------------------------------------------------------------

    print_metrics(
        "outputs/metrics/extractive_qa_trigger_test_metrics.json",
        "Trigger QA - test metrics",
    )

    print_metrics(
        "outputs/metrics/extractive_qa_main_arguments_test_metrics.json",
        "Main argument QA - test metrics",
    )

    print_metrics(
        "outputs/metrics/extractive_qa_subarguments_test_metrics.json",
        "Subargument QA - test metrics",
    )

    print("\nExtractive QA pipeline finished successfully.")


if __name__ == "__main__":
    main()
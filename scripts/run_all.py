# scripts/run_all.py

import getpass
import os
import subprocess
import sys
from pathlib import Path


def run_command(command, env=None):
    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80)

    subprocess.run(command, check=True, env=env)


def ask_yes_no(question, default=False):
    suffix = " [y/N]: " if not default else " [Y/n]: "
    answer = input(question + suffix).strip().lower()

    if answer == "":
        return default

    return answer in {"y", "yes", "t", "true", "1"}


def require_hf_token():
    hf_token = getpass.getpass("Paste your Hugging Face token: ").strip()

    if not hf_token:
        raise RuntimeError(
            "HF token is required to run this pipeline. "
            "Training was not started."
        )

    return hf_token


def main():
    print("== PHEE experiments pipeline started ==")

    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/timing").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    print("\nHugging Face token is required before training.")
    hf_token = require_hf_token()

    env["HF_TOKEN"] = hf_token
    env["UPLOAD_TO_HF"] = "true"

    print("Logging in to Hugging Face...")
    run_command(
        [
            sys.executable,
            "-c",
            (
                "from huggingface_hub import login; "
                "import os; "
                "login(token=os.environ['HF_TOKEN'])"
            ),
        ],
        env=env,
    )

    skip_train = ask_yes_no(
        "Do you want to skip training and use an existing model?",
        default=False,
    )

    run_test = ask_yes_no(
        "Do you want to run final evaluation on the test set?",
        default=False,
    )

    if not skip_train:
        print("\nStep 1: Training sequence labeling model")
        run_command(
            [
                sys.executable,
                "scripts/train_seq.py",
            ],
            env=env,
        )
    else:
        print("\nStep 1: Training skipped.")

    print("\nStep 2: Predicting on validation set")
    run_command(
        [
            sys.executable,
            "scripts/predict_seq.py",
            "--config",
            "configs/seq.yaml",
            "--split",
            "val",
            "--output_path",
            "outputs/predictions/seq_val.jsonl",
        ],
        env=env,
    )

    print("\nStep 3: Evaluating validation predictions")
    run_command(
        [
            sys.executable,
            "scripts/evaluate_predictions.py",
            "--predictions_path",
            "outputs/predictions/seq_val.jsonl",
            "--output_path",
            "outputs/metrics/seq_val_metrics.json",
        ],
        env=env,
    )

    if run_test:
        print("\nStep 4: Predicting on test set")
        run_command(
            [
                sys.executable,
                "scripts/predict_seq.py",
                "--config",
                "configs/seq.yaml",
                "--split",
                "test",
                "--output_path",
                "outputs/predictions/seq_test.jsonl",
            ],
            env=env,
        )

        print("\nStep 5: Evaluating test predictions")
        run_command(
            [
                sys.executable,
                "scripts/evaluate_predictions.py",
                "--predictions_path",
                "outputs/predictions/seq_test.jsonl",
                "--output_path",
                "outputs/metrics/seq_test_metrics.json",
            ],
            env=env,
        )
    else:
        print("\nTest evaluation skipped.")

    print("\n== Pipeline finished successfully ==")


if __name__ == "__main__":
    main()
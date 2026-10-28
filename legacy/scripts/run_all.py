# scripts/run_all.py

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path


MAIN_CONFIG = "configs/seq.yaml"
SUBARGUMENTS_CONFIG = "configs/seq_subarguments.yaml"

MAIN_VAL_PREDICTIONS = "outputs/predictions/seq_val.jsonl"
MAIN_TEST_PREDICTIONS = "outputs/predictions/seq_test.jsonl"

SUBARG_VAL_PREDICTIONS = "outputs/predictions/seq_subarguments_val.jsonl"
SUBARG_TEST_PREDICTIONS = "outputs/predictions/seq_subarguments_test.jsonl"

MAIN_VAL_METRICS = "outputs/metrics/seq_val_metrics.json"
MAIN_TEST_METRICS = "outputs/metrics/seq_test_metrics.json"

SUBARG_VAL_METRICS = "outputs/metrics/seq_subarguments_val_metrics.json"
SUBARG_TEST_METRICS = "outputs/metrics/seq_subarguments_test_metrics.json"


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


def prepare_output_dirs():
    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/timing").mkdir(parents=True, exist_ok=True)


def login_to_huggingface(env):
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


def train_main(env):
    print("\nStep MAIN-1: Training main sequence labeling model")
    run_command(
        [
            sys.executable,
            "scripts/train_seq.py",
        ],
        env=env,
    )


def predict_main(split, output_path, env):
    print(f"\nStep MAIN-2: Predicting main arguments on {split} set")
    run_command(
        [
            sys.executable,
            "scripts/predict_seq.py",
            "--config",
            MAIN_CONFIG,
            "--split",
            split,
            "--output_path",
            output_path,
        ],
        env=env,
    )


def evaluate_main(predictions_path, output_path, env):
    print("\nStep MAIN-3: Evaluating main argument predictions")
    run_command(
        [
            sys.executable,
            "scripts/evaluate_predictions.py",
            "--predictions_path",
            predictions_path,
            "--output_path",
            output_path,
        ],
        env=env,
    )


def train_subarguments(env):
    print("\nStep SUBARG-1: Training subargument sequence labeling model")
    run_command(
        [
            sys.executable,
            "scripts/train_seq_subarguments.py",
            "--config",
            SUBARGUMENTS_CONFIG,
        ],
        env=env,
    )


def predict_subarguments(split, output_path, env):
    print(f"\nStep SUBARG-2: Predicting subarguments on {split} set")
    run_command(
        [
            sys.executable,
            "scripts/predict_seq_subarguments.py",
            "--config",
            SUBARGUMENTS_CONFIG,
            "--split",
            split,
            "--output_path",
            output_path,
        ],
        env=env,
    )


def evaluate_subarguments(predictions_path, output_path, env):
    print("\nStep SUBARG-3: Evaluating subargument predictions")
    run_command(
        [
            sys.executable,
            "scripts/evaluate_predictions.py",
            "--predictions_path",
            predictions_path,
            "--output_path",
            output_path,
        ],
        env=env,
    )


def run_main_pipeline(skip_train, run_test, env):
    if not skip_train:
        train_main(env)
    else:
        print("\nMain model training skipped.")

    predict_main(
        split="val",
        output_path=MAIN_VAL_PREDICTIONS,
        env=env,
    )

    evaluate_main(
        predictions_path=MAIN_VAL_PREDICTIONS,
        output_path=MAIN_VAL_METRICS,
        env=env,
    )

    if run_test:
        predict_main(
            split="test",
            output_path=MAIN_TEST_PREDICTIONS,
            env=env,
        )

        evaluate_main(
            predictions_path=MAIN_TEST_PREDICTIONS,
            output_path=MAIN_TEST_METRICS,
            env=env,
        )
    else:
        print("\nMain test evaluation skipped.")


def run_subarguments_pipeline(skip_train, run_test, env):
    if not skip_train:
        train_subarguments(env)
    else:
        print("\nSubargument model training skipped.")

    predict_subarguments(
        split="val",
        output_path=SUBARG_VAL_PREDICTIONS,
        env=env,
    )

    evaluate_subarguments(
        predictions_path=SUBARG_VAL_PREDICTIONS,
        output_path=SUBARG_VAL_METRICS,
        env=env,
    )

    if run_test:
        predict_subarguments(
            split="test",
            output_path=SUBARG_TEST_PREDICTIONS,
            env=env,
        )

        evaluate_subarguments(
            predictions_path=SUBARG_TEST_PREDICTIONS,
            output_path=SUBARG_TEST_METRICS,
            env=env,
        )
    else:
        print("\nSubargument test evaluation skipped.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PHEE experiments pipeline."
    )

    parser.add_argument(
        "--target",
        choices=["all", "main", "subarguments"],
        default="all",
        help=(
            "Which part of the pipeline to run. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--method",
        choices=["seq"],
        default="seq",
        help=(
            "Which method to run. Currently only 'seq' is supported. "
            "This argument is kept for future methods such as extractive_qa or generative_qa."
        ),
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and use existing trained models.",
    )

    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip final evaluation on the test set.",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask interactively whether to skip training and run test evaluation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("== PHEE experiments pipeline started ==")

    prepare_output_dirs()

    env = os.environ.copy()

    skip_train = args.skip_train
    run_test = not args.no_test

    if args.interactive:
        skip_train = ask_yes_no(
            "Do you want to skip training and use an existing model?",
            default=skip_train,
        )

        run_test = ask_yes_no(
            "Do you want to run final evaluation on the test set?",
            default=run_test,
        )

    if not skip_train:
        login_to_huggingface(env)
    else:
        print("\nTraining skipped, Hugging Face login skipped.")

    if args.method != "seq":
        raise ValueError(
            f"Unsupported method: {args.method}. "
            "Currently only 'seq' is implemented."
        )

    if args.target in {"all", "main"}:
        print("\n" + "#" * 80)
        print("Running MAIN ARGUMENT pipeline")
        print("#" * 80)
        run_main_pipeline(
            skip_train=skip_train,
            run_test=run_test,
            env=env,
        )

    if args.target in {"all", "subarguments"}:
        print("\n" + "#" * 80)
        print("Running SUBARGUMENT pipeline")
        print("#" * 80)
        run_subarguments_pipeline(
            skip_train=skip_train,
            run_test=run_test,
            env=env,
        )

    print("\n== Pipeline finished successfully ==")


if __name__ == "__main__":
    main()
# liczenie wspolnych metryk
import argparse
import json
from pathlib import Path

from src.evaluation.span_metrics import compute_dataset_metrics


def load_jsonl(path: str):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--output_path", default=None)

    args = parser.parse_args()

    records = load_jsonl(args.predictions_path)
    metrics = compute_dataset_metrics(records)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
"""Aggregate all finished runs into tables and a plot.

    python scripts/aggregate_results.py --results_dir results --gpu_price 0.25

Writes results/tables/:
  comparison_test.md                  baseline -> improved version of every method
  main_test.md                        the main metrics (arguments, event core, arguments in
                                      the event) and the cost, mean ± std over seeds
  main_test.csv                       every metric of every row
  details_test.md                     triggers, sub-arguments, token-F1, time, GPU memory
  subsets_test.md                     single-event vs multi-event sentences
  labels_test.csv / labels_test.md    F1 per label (rare labels first)
  dependency_test.csv                 trigger / parent dependency of missed arguments
  quality_vs_cost_test.png            arguments-in-event F1 vs inference time per sentence
  checks.md                           what to look at (failed runs, epochs, formats)
and the same for the dev set (*_dev.*).

--gpu_price: the price of the rented GPU in USD per hour (vast.ai shows it for the
instance); the cost of a run = its wall clock x the price. Default: the GPU_PRICE
environment variable, else 0.25 (the price assumed for the thesis).
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phee.analysis import (  # noqa: E402
    aggregate,
    markdown_comparison,
    markdown_labels,
    markdown_details,
    markdown_main,
    markdown_subsets,
    plot_quality_cost,
    run_checks,
    write_csv,
)
from phee.io import read_yaml  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--matrix", default=str(ROOT / "configs" / "matrix.yaml"))
    parser.add_argument("--gpu_price", type=float, default=float(os.environ.get("GPU_PRICE", "0.25")),
                        help="GPU rental price in USD per hour")
    args = parser.parse_args()
    out = Path(args.results_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pairs = read_yaml(args.matrix).get("comparison", [])

    for split in ("test", "dev"):
        rows, label_rows, dependency_rows = aggregate(args.results_dir, split, args.gpu_price)
        if not rows:
            print(f"No finished runs with a {split} split in {args.results_dir}")
            continue
        write_csv(rows, out / f"main_{split}.csv")
        write_csv(label_rows, out / f"labels_{split}.csv")
        write_csv(dependency_rows, out / f"dependency_{split}.csv")
        (out / f"main_{split}.md").write_text(markdown_main(rows, args.gpu_price), encoding="utf-8")
        (out / f"details_{split}.md").write_text(markdown_details(rows, args.gpu_price), encoding="utf-8")
        (out / f"comparison_{split}.md").write_text(markdown_comparison(rows, pairs), encoding="utf-8")
        (out / f"subsets_{split}.md").write_text(markdown_subsets(rows), encoding="utf-8")
        (out / f"labels_{split}.md").write_text(markdown_labels(label_rows), encoding="utf-8")
        plot_quality_cost(rows, out / f"quality_vs_cost_{split}.png")
        print(f"{split}: {len(rows)} (experiment, variant) rows")
        if split == "test":
            print(markdown_comparison(rows, pairs))
            print(markdown_main(rows, args.gpu_price))

    issues = run_checks(args.results_dir)
    text = "# Checks of the finished runs (dev set)\n\n" + (
        "\n".join(f"- {issue}" for issue in issues) if issues else "Nothing suspicious found."
    ) + "\n"
    (out / "checks.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Tables in {out}/")


if __name__ == "__main__":
    main()

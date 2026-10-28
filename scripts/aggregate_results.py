"""Aggregate all finished runs into tables and a plot.

    python scripts/aggregate_results.py --results_dir results

Writes results/tables/:
  main_test.csv / main_test.md        F1 per level (mean ± std over seeds) and cost
  subsets_test.md                     single-event vs multi-event sentences
  labels_test.csv / labels_test.md    F1 per label (rare labels first)
  dependency_test.csv                 trigger / parent dependency of missed arguments
  quality_vs_cost_test.png            overall F1 vs inference time per sentence
and the same for the dev set (*_dev.*).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phee.analysis import (  # noqa: E402
    aggregate,
    markdown_labels,
    markdown_main,
    markdown_subsets,
    plot_quality_cost,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    out = Path(args.results_dir) / "tables"

    for split in ("test", "dev"):
        rows, label_rows, dependency_rows = aggregate(args.results_dir, split)
        if not rows:
            print(f"No finished runs with a {split} split in {args.results_dir}")
            continue
        write_csv(rows, out / f"main_{split}.csv")
        write_csv(label_rows, out / f"labels_{split}.csv")
        write_csv(dependency_rows, out / f"dependency_{split}.csv")
        (out / f"main_{split}.md").write_text(markdown_main(rows), encoding="utf-8")
        (out / f"subsets_{split}.md").write_text(markdown_subsets(rows), encoding="utf-8")
        (out / f"labels_{split}.md").write_text(markdown_labels(label_rows), encoding="utf-8")
        plot_quality_cost(rows, out / f"quality_vs_cost_{split}.png")
        print(f"{split}: {len(rows)} (experiment, variant) rows")
        if split == "test":
            print(markdown_main(rows))


if __name__ == "__main__":
    main()

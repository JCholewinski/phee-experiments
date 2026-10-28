"""Convert the revised PHEE data to the unified format.

    python scripts/prepare_data.py

Writes data/phee2/processed/{train,dev,test}.jsonl and prints conversion statistics.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phee.data import SPLITS, convert_split  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="data/phee2/source")
    parser.add_argument("--tokenized_dir", default="data/phee_v1_tokenized")
    parser.add_argument("--output_dir", default="data/phee2/processed")
    args = parser.parse_args()

    all_stats = {}
    for split in SPLITS:
        stats = convert_split(
            source_path=Path(args.source_dir) / f"{split}.json",
            tokenized_path=Path(args.tokenized_dir) / f"{split}.json",
            output_path=Path(args.output_dir) / f"{split}.jsonl",
        )
        all_stats[split] = stats
        print(f"{split}: {json.dumps(stats, sort_keys=True)}")

    with open(Path(args.output_dir) / "conversion_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

"""Run one experiment config with one seed.

    python scripts/run_experiment.py --config configs/seq/own_linear.yaml --seed 1

Results: results/<experiment name>/seed<seed>/  (summary.json, metrics/, predictions/,
training.json, history/). A finished run contains a DONE file and is skipped unless
--force is given.

Overrides (for quick tests):
    --set data.limit=50 --set training.max_epochs=1
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phee.io import load_config  # noqa: E402
from phee.utils import log  # noqa: E402

RUNNERS = {
    "seq": "phee.seq.runner",
    "eqa": "phee.eqa.runner",
    "gqa": "phee.gqa.runner",
    "llm": "phee.llm.runner",
}


def apply_override(config, assignment):
    key, _, raw = assignment.partition("=")
    value = yaml.safe_load(raw)
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--name", default=None, help="experiment name (default: config path)")
    parser.add_argument("--set", action="append", default=[], help="override, e.g. training.max_epochs=1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    for assignment in args.set:
        apply_override(config, assignment)
    if args.name:
        config["name"] = args.name
    elif not config.get("name"):
        parts = config_path.with_suffix("").parts
        config["name"] = "__".join(parts[1:] if parts[0] == "configs" else parts)

    run_dir = Path(args.output_dir) / config["name"] / f"seed{args.seed}"
    if (run_dir / "DONE").exists() and not args.force:
        log(f"skip {run_dir} (already done)")
        return
    log(f"run {config['name']} seed {args.seed} -> {run_dir}")

    import importlib

    runner = importlib.import_module(RUNNERS[config["method"]])
    try:
        summary = runner.run(config, args.seed, run_dir)
    except Exception:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "FAILED").write_text(traceback.format_exc())
        raise
    printable = {
        variant: {split: values["summary"] for split, values in splits.items()}
        for variant, splits in summary["variants"].items()
    }
    log("test summary:\n" + json.dumps(
        {v: {k: round(x, 4) for k, x in s.get("test", {}).items() if k.endswith("classification_f1") and "_event_" not in k}
         for v, s in printable.items()},
        indent=2,
    ))


if __name__ == "__main__":
    main()

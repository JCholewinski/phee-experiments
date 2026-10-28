"""Smoke test: every experiment config end-to-end on a few sentences with the tiny
random models from make_tiny_models.py (CPU is enough). Checks that training, early
stopping, prediction, every variant and the evaluation run without errors; the scores
of random models are meaningless.

    python scripts/smoke/make_tiny_models.py
    python scripts/smoke/run_smoke.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phee.io import load_config, read_yaml  # noqa: E402

TINY = {
    "dmis-lab/biobert-base-cased-v1.1": "bert",
    "thomas-sounack/Bio-ModernBERT-base": "modernbert",
    "razent/SciFive-base-PMC": "t5",
    "google/t5gemma-b-b-ul2": "t5gemma",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=".smoke/models")
    parser.add_argument("--output_dir", default=".smoke/results")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--only", nargs="*", default=None, help="substring filter on config paths")
    args = parser.parse_args()

    matrix = read_yaml(ROOT / "configs" / "matrix.yaml")
    configs = [path for group in matrix["groups"].values() for path in group["configs"]]
    if args.only:
        configs = [c for c in configs if any(o in c for o in args.only)]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    failures = []
    for config_path in configs:
        config = load_config(ROOT / config_path)
        overrides = [
            f"data.limit={args.limit}",
            "training.max_epochs=2",
            "training.patience=1",
            "training.batch_size=4",
            "training.eval_batch_size=8",
            "training.log_every=0",
        ]
        if config["method"] == "llm":
            overrides += [
                f"model.name={args.models}/llm",
                "model.backend=hf",
                "max_new_tokens=24",
                "shots=[0, 2]",
            ]
        else:
            tiny = TINY[config["model"]["name"]]
            overrides.append(f"model.name={args.models}/{tiny}")
        if config["method"] == "gqa":
            overrides += [
                "generation.stage1_max_new_tokens=24",
                "generation.stage2_max_new_tokens=12",
                "generation.num_beams=2",
            ]
        command = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config", config_path,
                   "--seed", "1", "--output_dir", args.output_dir, "--force"]
        for override in overrides:
            command += ["--set", override]
        print(f"\n######## {config_path}", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            failures.append(config_path)
    print("\nSmoke test:", "OK" if not failures else f"FAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

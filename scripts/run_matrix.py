"""Run groups of experiments from configs/matrix.yaml, every run in its own process.

    python scripts/run_matrix.py --groups baselines seq eqa gqa --seeds 1 2 3
    python scripts/run_matrix.py --groups llm            # LLM groups use their own seeds
    python scripts/run_matrix.py --list                  # show what would run

Finished runs (DONE file) are skipped, so the command can simply be started again
after an interruption. A failed run is logged and the next one starts.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phee.io import load_config, read_yaml  # noqa: E402


def experiment_name(config_path):
    config = load_config(ROOT / config_path)
    if config.get("name"):
        return config["name"]
    parts = Path(config_path).with_suffix("").parts
    return "__".join(parts[1:] if parts[0] == "configs" else parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/matrix.yaml")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--set", action="append", default=[], help="override passed to every run")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    args = parser.parse_args()

    matrix = read_yaml(ROOT / args.matrix)
    groups = args.groups or list(matrix["groups"])
    log_dir = ROOT / args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for group in groups:
        spec = matrix["groups"][group]
        seeds = spec.get("seeds") if spec.get("fixed_seeds") else (args.seeds or spec.get("seeds", [1, 2, 3]))
        for config_path in spec["configs"]:
            for seed in seeds:
                jobs.append((group, config_path, seed))

    for group, config_path, seed in jobs:
        name = experiment_name(config_path)
        done = (ROOT / args.output_dir / name / f"seed{seed}" / "DONE").exists()
        status = "done" if done else "todo"
        print(f"[{status}] {group:10s} {config_path:45s} seed {seed}")
    if args.list:
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    failures = []
    for group, config_path, seed in jobs:
        name = experiment_name(config_path)
        if (ROOT / args.output_dir / name / f"seed{seed}" / "DONE").exists():
            continue
        log_path = log_dir / f"{name}__seed{seed}.log"
        command = [
            sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
            "--config", config_path, "--seed", str(seed), "--output_dir", args.output_dir,
        ]
        for assignment in args.set:
            command += ["--set", assignment]
        print(f"\n=== {name} seed {seed} (log: {log_path})", flush=True)
        start = time.time()
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
            process.wait()
        minutes = (time.time() - start) / 60
        if process.returncode:
            failures.append((name, seed))
            print(f"!!! {name} seed {seed} FAILED after {minutes:.1f} min (see {log_path})", flush=True)
            if args.stop_on_error:
                break
        else:
            print(f"=== {name} seed {seed} finished in {minutes:.1f} min", flush=True)

    if failures:
        print("\nFailed runs:")
        for name, seed in failures:
            print(f"  {name} seed {seed}")
        sys.exit(1)


if __name__ == "__main__":
    main()

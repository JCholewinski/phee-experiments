"""Run groups of experiments from configs/matrix.yaml, every run in its own process.

    python scripts/run_matrix.py --groups baselines seq eqa gqa --seeds 1 2 3
    python scripts/run_matrix.py --groups llm            # LLM groups use their own seeds
    python scripts/run_matrix.py --list                  # show what would run

Finished runs (DONE file) are skipped, so the command can simply be started again
after an interruption. A failed run is logged and the next one starts.
A finished run whose saved config.json differs from the current config (the config file
was changed after the run) is run again; the old run directory is moved to
<output_dir>/_replaced/ first.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phee.io import experiment_config, read_yaml  # noqa: E402


def run_status(config_path, seed, output_dir, overrides):
    """(experiment name, run dir, status): todo | done | changed (finished with a config
    that differs from the current one)."""
    config = experiment_config(config_path, seed, overrides, root=ROOT)
    run_dir = ROOT / output_dir / config["name"] / f"seed{seed}"
    if not (run_dir / "DONE").exists():
        return config["name"], run_dir, "todo"
    saved_path = run_dir / "config.json"
    current = json.loads(json.dumps(config))
    if saved_path.exists() and json.loads(saved_path.read_text(encoding="utf-8")) != current:
        return config["name"], run_dir, "changed"
    return config["name"], run_dir, "done"


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
        _, _, status = run_status(config_path, seed, args.output_dir, args.set)
        print(f"[{status:7s}] {group:10s} {config_path:45s} seed {seed}")
    if args.list:
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    failures = []
    for group, config_path, seed in jobs:
        name, run_dir, status = run_status(config_path, seed, args.output_dir, args.set)
        if status == "done":
            continue
        if status == "changed":
            backup = ROOT / args.output_dir / "_replaced" / name / f"seed{seed}-{time.strftime('%Y%m%d-%H%M%S')}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(run_dir), str(backup))
            print(f"--- {name} seed {seed}: config changed since the run, old run moved to {backup}", flush=True)
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

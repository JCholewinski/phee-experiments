"""Bookkeeping shared by all experiment runners: run directories, saving predictions,
evaluation of every variant and the run summary."""

import json
import platform
import time
from pathlib import Path

from phee.data import load_samples
from phee.evaluation import evaluate
from phee.io import write_json, write_jsonl


def load_splits(config, splits=("train", "dev", "test")):
    data_dir = Path(config.get("data", {}).get("dir", "data/phee2/processed"))
    limit = config.get("data", {}).get("limit")
    return {
        split: load_samples(data_dir / f"{split}.jsonl", limit=limit)
        for split in splits
    }


def early_stopping_dev(config, data):
    """Dev sentences used for the per-epoch evaluation (early stopping). The full dev
    set by default; `early_stopping_dev_limit: N` uses N evenly spaced sentences to save
    time (final dev results are always computed on the full dev set)."""
    limit = config.get("early_stopping_dev_limit")
    dev = data["dev"]
    if not limit or limit >= len(dev):
        return dev
    step = len(dev) / limit
    return [dev[int(i * step)] for i in range(limit)]


def environment_info():
    info = {"python": platform.python_version(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import torch
        import transformers

        info["torch"] = torch.__version__
        info["transformers"] = transformers.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return info


class RunRecorder:
    """Collects predictions, metrics and costs of one (experiment, seed) run."""

    def __init__(self, run_dir, config, seed):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.seed = seed
        self.stages = []
        self.variants = {}
        write_json(config, self.run_dir / "config.json")

    def add_stage(self, report):
        report = dict(report)
        history = report.pop("history", None)
        if history is not None:
            write_json(history, self.run_dir / "history" / f"{report['name']}.json")
        self.stages.append(report)
        write_json(self.stages, self.run_dir / "training.json")

    def add_variant(self, variant, split, samples, predictions, inference=None):
        """Save predictions of one variant on one split and evaluate them."""
        write_jsonl(predictions, self.run_dir / "predictions" / f"{variant}__{split}.jsonl")
        metrics = evaluate(samples, predictions)
        write_json(metrics, self.run_dir / "metrics" / f"{variant}__{split}.json")
        entry = self.variants.setdefault(variant, {})
        entry[split] = {"summary": metrics["summary"]}
        if inference is not None:
            inference = dict(inference)
            inference["sentences"] = len(samples)
            if inference.get("seconds") is not None and samples:
                inference["seconds_per_sentence"] = inference["seconds"] / len(samples)
            entry[split]["inference"] = inference
        return metrics

    def finish(self, extra=None):
        train_seconds = sum(stage.get("train_seconds") or 0 for stage in self.stages)
        dev_eval_seconds = sum(stage.get("dev_eval_seconds") or 0 for stage in self.stages)
        peaks = [stage.get("peak_gpu_memory_gb") for stage in self.stages if stage.get("peak_gpu_memory_gb")]
        summary = {
            "experiment": self.config.get("name"),
            "method": self.config.get("method"),
            "seed": self.seed,
            "environment": environment_info(),
            "training": {
                "train_seconds": train_seconds,
                "dev_eval_seconds": dev_eval_seconds,
                "peak_gpu_memory_gb": max(peaks) if peaks else None,
                "stages": [
                    {key: stage.get(key) for key in (
                        "name", "best_epoch", "epochs_run", "best_dev_f1", "train_seconds",
                        "dev_eval_seconds", "train_examples", "peak_gpu_memory_gb", "parameters",
                    )}
                    for stage in self.stages
                ],
            },
            "variants": self.variants,
        }
        if extra:
            summary.update(extra)
        write_json(summary, self.run_dir / "summary.json")
        (self.run_dir / "DONE").write_text(json.dumps({"time": time.time()}))
        return summary

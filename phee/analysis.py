"""Aggregation of all runs: mean and standard deviation over seeds for every
(experiment, variant), plus per-label, subset, dependency and cost tables."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from phee.schema import MAIN_LABELS, SUB_LABELS

MAIN_METRICS = [
    "trigger_classification_f1",
    "trigger_identification_f1",
    "event_type_f1",
    "main_classification_f1",
    "main_classification_token_f1",
    "main_identification_f1",
    "sub_classification_f1",
    "sub_classification_token_f1",
    "arguments_classification_f1",
    "overall_classification_f1",
]
DEPENDENCY_KEYS = [
    "main_trigger_found_arg_found",
    "main_trigger_found_arg_missed",
    "main_trigger_missed_arg_found",
    "main_trigger_missed_arg_missed",
    "sub_parent_found_arg_found",
    "sub_parent_found_arg_missed",
    "sub_parent_missed_arg_found",
    "sub_parent_missed_arg_missed",
]
SUBSET_METRICS = [
    "single_event_main_classification_f1",
    "multi_event_main_classification_f1",
    "single_event_overall_classification_f1",
    "multi_event_overall_classification_f1",
]


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def collect_runs(results_dir):
    runs = []
    for summary_path in sorted(Path(results_dir).glob("*/seed*/summary.json")):
        run_dir = summary_path.parent
        if not (run_dir / "DONE").exists():
            continue
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        summary["_run_dir"] = str(run_dir)
        config_path = run_dir / "config.json"
        summary["_config"] = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        runs.append(summary)
    return runs


def _metrics_file(run, variant, split):
    path = Path(run["_run_dir"]) / "metrics" / f"{variant}__{split}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _has_triggers(run):
    """The LLM baseline (prompt of Sun et al. 2024) does not ask for triggers, so its
    trigger scores are not reported (they would be 0 by construction)."""
    return run.get("_config", {}).get("prompt_style") != "authors"


def aggregate(results_dir, split="test", price_per_hour=None):
    """Rows per (experiment, variant). `price_per_hour`: GPU rental price in USD per
    hour, used for the cost of a whole run (wall clock x price)."""
    runs = collect_runs(results_dir)
    groups = defaultdict(list)
    for run in runs:
        for variant, splits in run["variants"].items():
            if split in splits:
                groups[(run["experiment"], variant)].append((run, splits[split]))

    rows, label_rows, dependency_rows = [], [], []
    for (experiment, variant), entries in sorted(groups.items()):
        row = {"experiment": experiment, "variant": variant, "seeds": len(entries)}
        for metric in MAIN_METRICS + SUBSET_METRICS:
            mean, std = _mean_std([
                entry["summary"].get(metric) for run, entry in entries
                if _has_triggers(run) or not metric.startswith("trigger_")
            ])
            row[metric] = mean
            row[metric + "_std"] = std
        row["run_minutes"], row["run_minutes_std"] = _mean_std(
            [run["run_seconds"] / 60 for run, _ in entries if run.get("run_seconds") is not None]
        )
        row["run_cost_usd"] = (
            row["run_minutes"] / 60 * price_per_hour
            if row["run_minutes"] is not None and price_per_hour is not None else None
        )
        row["gpus"] = ", ".join(sorted({run.get("environment", {}).get("gpu") or "no GPU" for run, _ in entries}))
        train_seconds = [run["training"]["train_seconds"] for run, _ in entries]
        row["train_minutes"], row["train_minutes_std"] = _mean_std([s / 60 for s in train_seconds])
        row["train_peak_gpu_gb"], _ = _mean_std([run["training"].get("peak_gpu_memory_gb") for run, _ in entries])
        row["inference_ms_per_sentence"], row["inference_ms_per_sentence_std"] = _mean_std(
            [1000 * entry["inference"]["seconds_per_sentence"] for _, entry in entries
             if entry.get("inference", {}).get("seconds_per_sentence") is not None]
        )
        row["inference_peak_gpu_gb"], _ = _mean_std(
            [entry.get("inference", {}).get("peak_gpu_memory_gb") for _, entry in entries]
        )
        # LLM with vLLM: no measurable peak (memory reserved up front), weight size instead
        row["inference_model_weights_gb"], _ = _mean_std(
            [entry.get("inference", {}).get("model_weights_gb") for _, entry in entries]
        )
        row["invalid_json_rate"], _ = _mean_std(
            [entry.get("inference", {}).get("invalid_json_rate") for _, entry in entries]
        )
        row["epochs"] = ", ".join(
            "/".join(str(stage.get("best_epoch")) for stage in run["training"]["stages"])
            for run, _ in entries
        )
        rows.append(row)

        per_label = defaultdict(list)
        dependency = defaultdict(list)
        for run, _ in entries:
            metrics = _metrics_file(run, variant, split)
            if not metrics:
                continue
            for label in MAIN_LABELS + SUB_LABELS:
                values = metrics["per_label"].get(label)
                if values:
                    per_label[label].append((values["em"]["f1"], values["support"]))
            for key, value in metrics.get("dependency", {}).items():
                dependency[key].append(value)
        for label, values in per_label.items():
            mean, std = _mean_std([v[0] for v in values])
            label_rows.append({
                "experiment": experiment, "variant": variant, "label": label,
                "support": values[0][1], "f1": mean, "f1_std": std,
            })
        if dependency:
            dep_row = {"experiment": experiment, "variant": variant}
            for key in DEPENDENCY_KEYS:
                values = dependency.get(key, [])
                # a key missing in a run means 0 cases in that run
                dep_row[key] = sum(values) / len(entries) if values else 0.0
            found_missed = dep_row.get("main_trigger_found_arg_missed", 0)
            missed_missed = dep_row.get("main_trigger_missed_arg_missed", 0)
            total_missed = found_missed + missed_missed
            dep_row["share_of_missed_args_with_missed_trigger"] = missed_missed / total_missed if total_missed else None
            sub_found_missed = dep_row.get("sub_parent_found_arg_missed", 0)
            sub_missed_missed = dep_row.get("sub_parent_missed_arg_missed", 0)
            sub_total = sub_found_missed + sub_missed_missed
            dep_row["share_of_missed_subargs_with_missed_parent"] = sub_missed_missed / sub_total if sub_total else None
            dependency_rows.append(dep_row)
    return rows, label_rows, dependency_rows


def write_csv(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(mean, std=None, scale=100.0, digits=1):
    if mean is None:
        return "–"
    if std is None:
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def markdown_main(rows, price_per_hour=None):
    header = (
        "| Experiment | Variant | Seeds | Trigger F1 | Main F1 | Sub F1 | Overall F1 | "
        "Train min | Run min | Run $ | Inference ms/sent | GPU GB (train/inf) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in rows:
        inference_gpu = _fmt(row["inference_peak_gpu_gb"], None, scale=1)
        if row["inference_peak_gpu_gb"] is None and row.get("inference_model_weights_gb") is not None:
            inference_gpu = f"{row['inference_model_weights_gb']:.1f} (weights)"
        lines.append(
            f"| {row['experiment']} | {row['variant']} | {row['seeds']} | "
            f"{_fmt(row['trigger_classification_f1'], row['trigger_classification_f1_std'])} | "
            f"{_fmt(row['main_classification_f1'], row['main_classification_f1_std'])} | "
            f"{_fmt(row['sub_classification_f1'], row['sub_classification_f1_std'])} | "
            f"{_fmt(row['overall_classification_f1'], row['overall_classification_f1_std'])} | "
            f"{_fmt(row['train_minutes'], row['train_minutes_std'], scale=1)} | "
            f"{_fmt(row.get('run_minutes'), None, scale=1)} | "
            f"{_fmt(row.get('run_cost_usd'), None, scale=1, digits=2)} | "
            f"{_fmt(row['inference_ms_per_sentence'], None, scale=1)} | "
            f"{_fmt(row['train_peak_gpu_gb'], None, scale=1)} / {inference_gpu} |"
        )
    gpus = sorted({row.get("gpus") or "?" for row in rows})
    price = f"{price_per_hour:.2f} USD/h" if price_per_hour is not None else "no price given"
    note = (
        "\nEM-F1 [%], mean ± std over seeds. Trigger F1 is not reported for the LLM baseline "
        "(the prompt of Sun et al. 2024 does not ask for triggers).\n"
        "Run min / Run $: wall clock of the whole run (loading, training, dev evaluations, "
        f"prediction of all variants of the run) x GPU price ({price}).\n"
        "GPU GB: peak memory allocated by PyTorch. LLM rows with vLLM show the size of the "
        "model weights, because vLLM reserves a fixed share of the GPU for its KV cache.\n"
        f"GPU: {'; '.join(gpus)}"
        + (" -- WARNING: runs on different GPUs, times are not comparable.\n" if len(gpus) > 1 else "\n")
    )
    return header + "\n".join(lines) + "\n" + note


def markdown_subsets(rows):
    header = (
        "| Experiment | Variant | Main F1 single-event | Main F1 multi-event | "
        "Overall F1 single-event | Overall F1 multi-event |\n|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {row['experiment']} | {row['variant']} | "
        f"{_fmt(row['single_event_main_classification_f1'], row['single_event_main_classification_f1_std'])} | "
        f"{_fmt(row['multi_event_main_classification_f1'], row['multi_event_main_classification_f1_std'])} | "
        f"{_fmt(row['single_event_overall_classification_f1'], row['single_event_overall_classification_f1_std'])} | "
        f"{_fmt(row['multi_event_overall_classification_f1'], row['multi_event_overall_classification_f1_std'])} |"
        for row in rows
    ]
    return header + "\n".join(lines) + "\n"


def markdown_labels(label_rows):
    """Rows: (experiment, variant); columns: labels from the rarest (test support)."""
    supports = {}
    values = defaultdict(dict)
    order_rows = []
    for row in label_rows:
        key = (row["experiment"], row["variant"])
        if key not in values:
            order_rows.append(key)
        values[key][row["label"]] = row["f1"]
        supports[row["label"]] = row["support"]
    labels = sorted(supports, key=lambda label: supports[label])
    header = "| Experiment | Variant | " + " | ".join(f"{label} ({supports[label]})" for label in labels) + " |\n"
    header += "|---|---|" + "---|" * len(labels) + "\n"
    lines = [
        f"| {experiment} | {variant} | "
        + " | ".join(_fmt(values[(experiment, variant)].get(label)) for label in labels) + " |"
        for experiment, variant in order_rows
    ]
    note = "\nEM-F1 [%], mean over seeds (std in labels_*.csv); in brackets: number of test items.\n"
    return header + "\n".join(lines) + "\n" + note


def plot_quality_cost(rows, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    points = [r for r in rows if r.get("inference_ms_per_sentence") and r.get("overall_classification_f1") is not None]
    if not points:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in points:
        ax.errorbar(
            row["inference_ms_per_sentence"], 100 * row["overall_classification_f1"],
            yerr=100 * (row.get("overall_classification_f1_std") or 0), fmt="o", capsize=3,
        )
        ax.annotate(f"{row['experiment']}\n{row['variant']}", (row["inference_ms_per_sentence"], 100 * row["overall_classification_f1"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Inference time per sentence on the test set [ms] (log scale)")
    ax.set_ylabel("Overall classification EM-F1 [%]")
    ax.set_title("Quality vs. inference cost (mean ± std over seeds)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def _select(rows, spec):
    """The row of `spec` = {experiment, variant (part of the name, optional)}; None if
    the run is missing or the choice is ambiguous."""
    found = [r for r in rows if r["experiment"] == spec["experiment"]
             and (spec.get("variant") is None or spec["variant"] in r["variant"])]
    return found[0] if len(found) == 1 else None


def markdown_comparison(rows, pairs):
    """Version A -> version B of every method (pairs from configs/matrix.yaml)."""
    columns = [
        ("Trigger F1", "trigger_classification_f1", 100.0, 1),
        ("Main F1", "main_classification_f1", 100.0, 1),
        ("Sub F1", "sub_classification_f1", 100.0, 1),
        ("Main F1 multi-event", "multi_event_main_classification_f1", 100.0, 1),
        ("Run $", "run_cost_usd", 1.0, 2),
    ]
    header = "| Method | A | B | " + " | ".join(name for name, *_ in columns) + " |\n"
    header += "|---|---|---|" + "---|" * len(columns) + "\n"
    lines = []
    for pair in pairs:
        a, b = _select(rows, pair["a"]), _select(rows, pair["b"])
        names = [spec["experiment"] + (f" ({spec['variant']})" if spec.get("variant") else "")
                 for spec in (pair["a"], pair["b"])]
        if a is None or b is None:
            lines.append(f"| {pair['method']} | {names[0]} | {names[1]} | "
                         + " | ".join("not run yet" for _ in columns) + " |")
            continue
        cells = []
        for _, key, scale, digits in columns:
            va, vb = a.get(key), b.get(key)
            if va is None or vb is None:
                cells.append(f"{_fmt(va, None, scale, digits)} → {_fmt(vb, None, scale, digits)}")
            else:
                cells.append(f"{va * scale:.{digits}f} → {vb * scale:.{digits}f} ({(vb - va) * scale:+.{digits}f})")
        lines.append(f"| {pair['method']} [seeds {a['seeds']}/{b['seeds']}] | {names[0]} | {names[1]} | "
                     + " | ".join(cells) + " |")
    note = ("\nEM-F1 [%], mean over seeds: A → B (difference). A = baseline, B = improved version; "
            "last LLM row: demonstrations from all 2898 training sentences → from 50 random ones. "
            "Run $: the whole run (all its variants, e.g. 0- and 5-shot of an LLM run).\n")
    return header + "\n".join(lines) + "\n" + note


def run_checks(results_dir, split="dev"):
    """What to look at after the test run: failed runs, stages that may need more epochs,
    suspiciously low scores, LLM answers in a wrong format, generations cut off."""
    issues = []
    for failed in sorted(Path(results_dir).glob("*/seed*/FAILED")):
        lines = [line for line in failed.read_text(encoding="utf-8").splitlines() if line.strip()]
        issues.append(f"FAILED {failed.parent.parent.name} {failed.parent.name}: {lines[-1] if lines else ''}")
    for run in collect_runs(results_dir):
        name = f"{run['experiment']} {Path(run['_run_dir']).name}"
        for stage in run["training"]["stages"]:
            best, ran = stage.get("best_epoch"), stage.get("epochs_run")
            if best and ran and best == ran:
                issues.append(f"{name}: stage {stage['name']} was best in its last epoch ({best}), "
                              "so max_epochs stopped it; the model might still improve")
            if stage.get("best_dev_f1") is not None and stage["best_dev_f1"] < 0.2:
                issues.append(f"{name}: stage {stage['name']} best dev F1 only {stage['best_dev_f1']:.3f}")
        for variant, splits in run["variants"].items():
            entry = splits.get(split)
            if not entry:
                continue
            main = entry["summary"].get("main_classification_f1")
            if main is not None and main < 0.3:
                issues.append(f"{name} {variant}: main argument F1 on {split} only {100 * main:.1f}")
            inference = entry.get("inference", {})
            rate = inference.get("invalid_json_rate")
            if rate and rate > 0.02:
                issues.append(f"{name} {variant}: {100 * rate:.1f}% of the answers on {split} not in the expected format")
            sentences = inference.get("sentences") or 0
            for key in ("stage1_truncated_outputs", "stage2_truncated_outputs"):
                count = inference.get(key) or 0
                if sentences and count > 0.01 * sentences:
                    issues.append(f"{name} {variant}: {count} outputs on {split} cut off by the length limit ({key})")
    return issues

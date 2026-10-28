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
        runs.append(summary)
    return runs


def _metrics_file(run, variant, split):
    path = Path(run["_run_dir"]) / "metrics" / f"{variant}__{split}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def aggregate(results_dir, split="test"):
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
            mean, std = _mean_std([entry["summary"].get(metric) for _, entry in entries])
            row[metric] = mean
            row[metric + "_std"] = std
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
            for key, values in sorted(dependency.items()):
                dep_row[key] = statistics.mean(values)
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


def markdown_main(rows):
    header = (
        "| Experiment | Variant | Seeds | Trigger F1 | Main F1 | Sub F1 | Overall F1 | "
        "Train min | Inference ms/sent | GPU GB (train/inf) |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['variant']} | {row['seeds']} | "
            f"{_fmt(row['trigger_classification_f1'], row['trigger_classification_f1_std'])} | "
            f"{_fmt(row['main_classification_f1'], row['main_classification_f1_std'])} | "
            f"{_fmt(row['sub_classification_f1'], row['sub_classification_f1_std'])} | "
            f"{_fmt(row['overall_classification_f1'], row['overall_classification_f1_std'])} | "
            f"{_fmt(row['train_minutes'], row['train_minutes_std'], scale=1)} | "
            f"{_fmt(row['inference_ms_per_sentence'], None, scale=1)} | "
            f"{_fmt(row['train_peak_gpu_gb'], None, scale=1)} / {_fmt(row['inference_peak_gpu_gb'], None, scale=1)} |"
        )
    return header + "\n".join(lines) + "\n"


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
    by_label = defaultdict(dict)
    columns = []
    supports = {}
    for row in label_rows:
        column = f"{row['experiment']} [{row['variant']}]"
        if column not in columns:
            columns.append(column)
        by_label[row["label"]][column] = _fmt(row["f1"], row["f1_std"])
        supports[row["label"]] = row["support"]
    order = sorted(by_label, key=lambda label: supports.get(label, 0))
    header = "| Label | Test support | " + " | ".join(columns) + " |\n"
    header += "|---|---|" + "---|" * len(columns) + "\n"
    lines = [
        f"| {label} | {supports.get(label, '')} | " + " | ".join(by_label[label].get(c, "–") for c in columns) + " |"
        for label in order
    ]
    return header + "\n".join(lines) + "\n"


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

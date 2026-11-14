import json
from pathlib import Path

import yaml


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base, override):
    """Recursively merge `override` into a copy of `base`."""
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    """Load a YAML config. A config may name a parent with `inherit: other.yaml`
    (path relative to the config file); the child overrides the parent."""
    path = Path(path)
    config = read_yaml(path) or {}
    parent = config.pop("inherit", None)
    if parent:
        parent_config = load_config(path.parent / parent)
        config = deep_update(parent_config, config)
    return config


def apply_override(config, assignment):
    """`key.sub=value` (value parsed as YAML), e.g. training.max_epochs=1."""
    key, _, raw = assignment.partition("=")
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # YAML reads "1e-3" (no dot) as a string
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def experiment_config(config_path, seed, overrides=(), name=None, root=None):
    """The config of one run exactly as run_experiment.py uses it (and saves it in
    config.json): inheritance, --set overrides, the experiment name from the config
    path (relative, e.g. configs/seq/own_crf.yaml; read from `root` if given),
    `data.train_sample.seed: run` replaced by the run seed."""
    config = load_config(Path(root) / config_path if root else config_path)
    for assignment in overrides:
        apply_override(config, assignment)
    sample = (config.get("data") or {}).get("train_sample")
    if sample and sample.get("seed") == "run":
        sample["seed"] = seed   # a different random subset of train for every seed
    if name:
        config["name"] = name
    elif not config.get("name"):
        parts = Path(config_path).with_suffix("").parts
        config["name"] = "__".join(parts[1:] if parts[0] == "configs" else parts)
    return config

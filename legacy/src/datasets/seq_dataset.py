import json
from datasets import Dataset

def load_dataset(path):
    with open(path) as f:
        data = json.load(f)
    return data

def build_dataset(data, convert_fn):
    processed = [convert_fn(x) for x in data]
    return Dataset.from_list(processed)
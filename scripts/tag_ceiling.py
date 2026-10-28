"""Upper bound of the sequence labeling tag schemes: the score of a perfect tagger
(gold tags decoded back to spans). Shows how much the "one tag per token" assumption
and the authors' duplication / single test copy cost, independently of any model.

    python scripts/tag_ceiling.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phee.data import load_samples  # noqa: E402
from phee.evaluation import evaluate  # noqa: E402
from phee.io import write_json  # noqa: E402
from phee.seq.tagging import (  # noqa: E402
    authors_event_tags,
    authors_tags_to_entities,
    bio_to_entities,
    entity_to_prediction,
    own_main_tags,
    own_sub_tags,
)


def two_level(sample):
    predictions = []
    for tags in (own_main_tags(sample)[0], own_sub_tags(sample)[0]):
        for start, end, entity in bio_to_entities(tags):
            prediction = entity_to_prediction(sample, start, end, entity)
            if prediction:
                predictions.append(prediction)
    return predictions


def authors_flat(sample):
    # at test time the authors keep one copy: the event with the most annotations
    def size(event):
        return len(event["trigger"]) + sum(len(v) for v in event["args"].values()) + sum(len(v) for v in event["subargs"].values())

    event = max(sample["events"], key=size)
    predictions = []
    for start, end, entity in authors_tags_to_entities(authors_event_tags(sample, event)):
        prediction = entity_to_prediction(sample, start, end, entity)
        if prediction:
            predictions.append(prediction)
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/phee2/processed/test.jsonl")
    parser.add_argument("--output", default="results/tables/tag_ceiling.json")
    args = parser.parse_args()
    samples = load_samples(args.data)
    report = {}
    for name, function in (("two_level", two_level), ("authors_flat", authors_flat)):
        metrics = evaluate(samples, [{"id": s["id"], "pred": function(s)} for s in samples])
        report[name] = {
            "summary": metrics["summary"],
            "per_label_f1": {label: values["em"]["f1"] for label, values in metrics["per_label"].items()},
        }
        print(name, json.dumps({k: round(100 * v, 1) for k, v in metrics["summary"].items()}, indent=1))
    write_json(report, args.output)


if __name__ == "__main__":
    main()

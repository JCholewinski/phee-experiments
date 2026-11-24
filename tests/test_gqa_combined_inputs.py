"""Generative QA stage 2 trained on combined inputs (train_inputs: combined)."""

from phee.gqa.runner import _stage2_training_items
from phee.schema import SUB_LABELS_BY_PARENT


class FakeTokenizer:
    def __call__(self, text, max_length=None, truncation=None):
        return {"input_ids": text}


class FakeSegments:
    def encode(self, segments, max_length):
        return segments


def _sample():
    sub = {label: [] for labels in SUB_LABELS_BY_PARENT.values() for label in labels}
    sub["TREATMENT_DRUG"] = [{"text": "insulin"}]
    sub["TREATMENT_DOSAGE"] = [{"text": "10 U"}]
    event = {
        "event_type": "ADE",
        "args": {"SUBJECT": [], "TREATMENT": [{"text": "insulin therapy"}], "EFFECT": [{"text": "lipoatrophy"}]},
        "subargs": sub,
    }
    return {"context": "insulin therapy 10 U caused lipoatrophy", "events": [event]}


def _items(predicted, include_gold):
    return _stage2_training_items(
        [_sample()], FakeTokenizer(), FakeSegments(), "current",
        {"max_source_length": 256, "max_target_length": 64},
        predicted_mains=predicted, include_gold=include_gold,
    )


def test_combined_inputs_keep_gold_add_inexact_and_false_arguments():
    n = len(SUB_LABELS_BY_PARENT["TREATMENT"])
    exact = {"label": "TREATMENT", "text": "insulin therapy", "event_type": "ADE"}
    inexact = {"label": "TREATMENT", "text": "insulin", "event_type": "ADE"}
    false = {"label": "TREATMENT", "text": "glucose", "event_type": "ADE"}

    gold_only = _items(None, False)
    assert len(gold_only) == n
    # predicted only: the gold argument stage 1 missed is not asked about at all
    assert len(_items([[false]], False)) == n
    combined = _items([[exact, inexact, false]], True)
    assert len(combined) == 3 * n                     # gold + inexact + false; exact = duplicate
    glucose = [i for i in combined if '"glucose"' in i["input_ids"]]
    assert len(glucose) == n and all("none" in str(i["labels"]).lower() for i in glucose)
    inexact_items = [i for i in combined if '"insulin"' in i["input_ids"]]       # the question names "insulin"
    assert len(inexact_items) == n
    assert any("10 U" in str(i["labels"]) for i in inexact_items)   # answers of the matched gold event

"""Tests of the model-independent logic (no torch needed).

    python -m pytest tests -q        or        python tests/test_core.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phee.data import gold_predictions, load_samples  # noqa: E402
from phee.evaluation import evaluate, score_families  # noqa: E402
from phee.gqa.format import (  # noqa: E402
    parse_stage1,
    parse_stage2,
    segments_text,
    stage1_segments,
    stage2_segments,
)
from phee.llm.bm25 import BM25  # noqa: E402
from phee.llm.prompts import extract_json, gold_json, to_predictions  # noqa: E402
from phee.normalize import normalize_text  # noqa: E402
from phee.seq.tagging import (  # noqa: E402
    authors_event_tags,
    authors_tags_to_entities,
    bio_to_entities,
    entity_to_prediction,
    own_main_tags,
    own_sub_tags,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "phee2" / "processed"
SAMPLES = load_samples(DATA / "test.jsonl")


def test_normalization_matches_authors():
    assert normalize_text("A 52-year-old Black woman") == "52 year old black woman"
    assert normalize_text("50% solution of dextrose") == "50 solution of dextrose"
    assert normalize_text("  The   patient ") == "patient"
    assert normalize_text("an") == ""


def test_gold_predictions_score_one():
    records = [{"id": s["id"], "pred": gold_predictions(s)} for s in SAMPLES]
    summary = evaluate(SAMPLES, records)["summary"]
    assert all(abs(value - 1.0) < 1e-9 for value in summary.values()), summary


def test_empty_predictions_score_zero():
    summary = evaluate(SAMPLES, [{"id": s["id"], "pred": []} for s in SAMPLES])["summary"]
    assert summary["overall_classification_f1"] == 0.0


def test_unique_pairs_and_no_event_type_for_arguments():
    sample = next(s for s in SAMPLES if s["is_multi_event"])
    predictions = gold_predictions(sample)
    # duplicates and wrong event types on arguments do not change the score
    noisy = predictions + [dict(p, event_type="PTE") for p in predictions if p["label"] != "TRIGGER"]
    score = score_families([sample], [{"id": sample["id"], "pred": noisy}], ["arguments_classification"])
    assert score["f1"] == 1.0


def test_two_level_tags_roundtrip_on_simple_sentence():
    sample = next(
        s for s in SAMPLES
        if not s["is_multi_event"] and not s["events"][0]["subargs"]["SUBJECT_DISORDER"]
        and not s["events"][0]["subargs"]["TREATMENT_DISORDER"]
    )
    predictions = []
    for tags in (own_main_tags(sample)[0], own_sub_tags(sample)[0]):
        for start, end, entity in bio_to_entities(tags):
            predictions.append(entity_to_prediction(sample, start, end, entity))
    score = score_families([sample], [{"id": sample["id"], "pred": predictions}],
                           ["trigger_classification", "main_classification"])
    assert score["f1"] == 1.0, (sample["id"], score)


def test_authors_tags_decode():
    sample = SAMPLES[0]
    tags = authors_event_tags(sample, sample["events"][0])
    entities = authors_tags_to_entities(tags)
    assert entities and all(start <= end for start, end, _ in entities)


def test_gqa_stage1_roundtrip_both_formats():
    for target_format, split_values in (("own", False), ("authors", True)):
        records = []
        for sample in SAMPLES:
            text = segments_text(stage1_segments(sample, target_format))
            predictions = []
            for event in parse_stage1(text, split_values=split_values):
                for trigger in event["trigger"]:
                    predictions.append({"label": "TRIGGER", "event_type": event["event_type"], "text": trigger})
                for label, values in event["args"].items():
                    predictions.extend({"label": label, "text": value} for value in values)
            records.append({"id": sample["id"], "pred": predictions})
        score = score_families(SAMPLES, records, ["trigger_classification", "main_classification"])
        # the authors' format splits on ";" which may cut a few spans containing ";"
        assert score["f1"] > (0.999 if target_format == "own" else 0.99), (target_format, score)


def test_gqa_stage2_answers():
    assert parse_stage2(segments_text(stage2_segments([]))) == []
    assert parse_stage2(segments_text(stage2_segments(["50 mg", "oral"]))) == ["50 mg", "oral"]


def test_llm_json_roundtrip():
    records = []
    for sample in SAMPLES:
        text = "```json\n" + gold_json(sample) + "\n```"
        predictions, valid = to_predictions(extract_json(text))
        assert valid
        records.append({"id": sample["id"], "pred": predictions})
    summary = evaluate(SAMPLES, records)["summary"]
    assert summary["overall_classification_f1"] == 1.0
    assert to_predictions(extract_json("no json here")) == ([], False)
    assert extract_json('<think>x</think> {"events": [],}')["events"] == []


def test_bm25_finds_identical_sentence():
    documents = [s["context"] for s in SAMPLES[:200]]
    retriever = BM25(documents)
    assert retriever.top(documents[17], 1) == [17]


def test_truncated_stage1_output():
    from phee.gqa.format import parse_stage1

    events = parse_stage1("[ADE] rash [Subject] 45-year-old woman [Eff")
    assert events[0]["args"]["SUBJECT"] == ["45-year-old woman"]


def test_no_whitespace_tokens():
    for split in ("train", "dev", "test"):
        samples = load_samples(DATA / f"{split}.jsonl")
        assert all(token.strip() for sample in samples for token in sample["tokens"])


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"ok   {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    sys.exit(1 if failures else 0)

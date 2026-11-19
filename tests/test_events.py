"""Event metrics (event core, arguments in the event) and the grouping rule of SL."""

import copy

from phee.data import gold_predictions, load_samples
from phee.evaluation import evaluate
from phee.events import assign_by_nearest_trigger, event_items, events_from_items, gold_events

TRAIN = "data/phee2/processed/train.jsonl"


def _sample(prefix):
    return next(s for s in load_samples(TRAIN) if s["context"].startswith(prefix))


def test_gold_is_perfect():
    samples = [s for s in load_samples(TRAIN) if s["is_multi_event"]][:30]
    summary = evaluate(samples, [{"id": s["id"], "pred": gold_predictions(s)} for s in samples])["summary"]
    assert summary["event_core_f1"] == 1.0
    assert summary["event_arguments_f1"] == 1.0


def test_items_without_events_count_as_no_event():
    sample = _sample("Atrial fibrillation was induced by diltiazem")
    pred = [{k: v for k, v in item.items() if k not in ("event", "event_type")} for item in gold_predictions(sample)]
    summary = evaluate([sample], [{"id": sample["id"], "pred": pred}])["summary"]
    assert summary["main_classification_f1"] == 1.0
    assert summary["event_core_f1"] == 0.0 and summary["event_arguments_f1"] == 0.0


def test_wrong_linking_is_caught():
    """Effects swapped between the two events: the argument metric of the PHEE paper does
    not notice it, arguments in the event do; the cores (type, drug) stay right."""
    sample = _sample("Atrial fibrillation was induced by diltiazem")
    pred = gold_predictions(sample)
    effects = [item for item in pred if item["label"] == "EFFECT"]
    assert len({item["event"] for item in effects}) == 2
    effects[0]["event"], effects[1]["event"] = effects[1]["event"], effects[0]["event"]
    summary = evaluate([sample], [{"id": sample["id"], "pred": pred}])["summary"]
    assert summary["main_classification_f1"] == 1.0
    assert summary["event_core_f1"] == 1.0
    assert summary["event_arguments_f1"] < 1.0


def test_same_core_events_are_merged():
    """Two gold events with the same (type, drug) and one predicted event with both
    effects: correct (one record per drug and type)."""
    sample = copy.deepcopy(load_samples(TRAIN)[0])
    event = sample["events"][0]
    event["subargs"]["TREATMENT_DRUG"] = [{"text": "pentamidine"}]
    event["args"] = {"EFFECT": [{"text": "cardiotoxicity"}], "TREATMENT": [{"text": "pentamidine"}]}
    event["subargs"] = {"TREATMENT_DRUG": [{"text": "pentamidine"}]}
    second = copy.deepcopy(event)
    second["args"]["EFFECT"] = [{"text": "heart block"}]
    sample["events"] = [event, second]
    pred = [
        {"label": "TREATMENT_DRUG", "text": "pentamidine", "event": 0, "event_type": event["event_type"]},
        {"label": "TREATMENT", "text": "pentamidine", "event": 0, "event_type": event["event_type"]},
        {"label": "EFFECT", "text": "cardiotoxicity", "event": 0, "event_type": event["event_type"]},
        {"label": "EFFECT", "text": "heart block", "event": 0, "event_type": event["event_type"]},
    ]
    assert event_items(gold_events(sample)) == event_items(events_from_items(pred))


def test_nearest_trigger_grouping():
    items = [
        {"label": "EFFECT", "text": "Atrial fibrillation", "tok_start": 0, "tok_end": 1},
        {"label": "TRIGGER", "text": "induced", "event_type": "ADE", "tok_start": 3, "tok_end": 3},
        {"label": "TREATMENT", "text": "diltiazem", "tok_start": 5, "tok_end": 5},
        {"label": "TREATMENT_DRUG", "text": "diltiazem", "tok_start": 5, "tok_end": 5},
        {"label": "TREATMENT", "text": "verapamil", "tok_start": 10, "tok_end": 10},
        {"label": "TREATMENT_DRUG", "text": "verapamil", "tok_start": 10, "tok_end": 10},
        {"label": "TRIGGER", "text": "induced", "event_type": "ADE", "tok_start": 11, "tok_end": 11},
        {"label": "EFFECT", "text": "syncope", "tok_start": 12, "tok_end": 12},
    ]
    events = events_from_items(assign_by_nearest_trigger(items))
    assert [dict(e["subargs"])["TREATMENT_DRUG"] for e in events] == [["diltiazem"], ["verapamil"]]
    assert [dict(e["args"])["EFFECT"] for e in events] == [["Atrial fibrillation"], ["syncope"]]


def test_no_trigger_no_event():
    items = [{"label": "EFFECT", "text": "rash", "tok_start": 0, "tok_end": 0}]
    assert events_from_items(assign_by_nearest_trigger(items)) == []


def test_eqa_records_keep_events():
    from phee.eqa.runner import _records, _trigger_key
    sample = {"id": "x"}
    t1 = {"label": "TRIGGER", "event_type": "ADE", "text": "induced", "start": 26, "end": 33}
    t2 = {"label": "TRIGGER", "event_type": "ADE", "text": "induced", "start": 80, "end": 87}
    mains = [{"label": "TREATMENT", "text": "diltiazem", "trigger_key": _trigger_key(t1), "event_type": "ADE"},
             {"label": "TREATMENT", "text": "verapamil", "trigger_key": _trigger_key(t2), "event_type": "ADE"}]
    subs = [{"label": "TREATMENT_DRUG", "text": "verapamil", "trigger_key": _trigger_key(t2), "event_type": "ADE"}]
    record = _records([sample], [[t1, t2]], [mains], [subs], with_events=True)[0]
    events = events_from_items(record["pred"])
    assert len(events) == 2
    assert dict(events[1]["args"])["TREATMENT"] == ["verapamil"]
    assert dict(events[1]["subargs"])["TREATMENT_DRUG"] == ["verapamil"]


def test_gqa_stage2_answers_go_to_the_asking_events():
    from phee.gqa.runner import _predict_stage2, _records, _stage1_predictions
    sample = {"id": "x", "context": "Atrial fibrillation was induced by diltiazem and verapamil induced syncope."}
    text = "[ADE] induced [Treatment] diltiazem [Effect] Atrial fibrillation [ADE] induced [Treatment] verapamil [Effect] syncope"
    triggers, mains = _stage1_predictions([text], split_values=False)

    def generate(model, inputs, samples, max_new_tokens, constraint):
        return ["diltiazem" if "diltiazem" in source.split("context:")[0] else "verapamil" for source in inputs]

    # own questions name the treatment: every answer goes to its own event
    subs = _predict_stage2(None, generate, [sample], mains, "current", {"stage2_max_new_tokens": 8}, None)
    events = events_from_items(_records([sample], triggers, mains, subs)[0]["pred"])
    assert [dict(e["subargs"]).get("TREATMENT_DRUG") for e in events] == [["diltiazem"], ["verapamil"]]
    # the authors' question ("Treatment.Drug in adverse event") is one for both events
    subs = _predict_stage2(None, generate, [sample], mains, "authors", {"stage2_max_new_tokens": 8}, None)
    events = events_from_items(_records([sample], triggers, mains, subs)[0]["pred"])
    assert all(dict(e["subargs"]).get("TREATMENT_DRUG") == ["verapamil"] for e in events)

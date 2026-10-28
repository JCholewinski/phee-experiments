"""Extractive QA pipeline experiment.

Stages (each with its own model, trained with early stopping on its own dev task):
  1. event type classifier (multi-label ADE / PTE)
  2. trigger QA: one question per predicted event type
  3. main-argument QA: one question per (predicted trigger, SUBJECT / TREATMENT / EFFECT)
  4. sub-argument QA: one question per (predicted main argument, sub-argument label)

Variants:
  templates     authors | current | definitions          (phee.eqa.templates)
  train_inputs  gold      stages 3 and 4 are trained on questions built from gold
                          triggers / main arguments
                predicted stages 3 and 4 are trained on questions built from the
                          previous stage's predictions on the training set (the existing
                          implementation; errors of earlier stages are seen in training)
At test time questions are always built from predictions (realistic pipeline).

Config (configs/eqa/*.yaml):
  method: eqa
  templates: current
  train_inputs: predicted
  model: {name, max_length: 384, stride: 128}
  event_type: {threshold: 0.5, force_one: true, max_length: 256}
  answers:
    trigger: {max_answer_length, max_spans, use_null, null_threshold}
    main:    {...}
    sub:     {...}
  training: common settings;  stage_training: {event_type|trigger|main|sub: overrides}
"""

from phee.eqa import event_type as event_type_stage
from phee.eqa.qa import accepts_token_type_ids, make_collate, predict_answers, train_features
from phee.eqa.templates import main_question, sub_question, trigger_question
from phee.evaluation import score_families
from phee.experiment import RunRecorder, early_stopping_dev, load_splits
from phee.hf import count_parameters, load_tokenizer
from phee.io import deep_update
from phee.schema import MAIN_LABELS, SUB_LABELS_BY_PARENT
from phee.training import train_model, training_settings
from phee.utils import GpuMemory, Timer, cleanup, log, set_seed

DEFAULT_ANSWERS = {
    "trigger": {"max_answer_length": 8, "max_spans": 1, "use_null": False, "null_threshold": 0.0},
    "main": {"max_answer_length": 60, "max_spans": 3, "use_null": True, "null_threshold": 0.0},
    "sub": {"max_answer_length": 50, "max_spans": 3, "use_null": True, "null_threshold": 0.0},
}


def _distance(a_start, a_end, b_start, b_end):
    if a_end <= b_start:
        return b_start - a_end
    if b_end <= a_start:
        return a_start - b_end
    return 0


def _example(sample, question, answer=None):
    return {
        "sample_id": sample["id"],
        "question": question,
        "context": sample["context"],
        "answer_start": answer["start"] if answer else None,
        "answer_text": answer["text"] if answer else None,
    }


def _unique_examples(examples):
    seen = set()
    result = []
    for example in examples:
        key = (example["sample_id"], example["question"], example["answer_start"], example["answer_text"])
        if key not in seen:
            seen.add(key)
            result.append(example)
    return result


def _answered(sample, question, spans):
    if spans:
        return [_example(sample, question, span) for span in spans]
    return [_example(sample, question)]


# ----------------------------------------------------------------- training examples

def trigger_training_examples(samples, templates):
    examples = []
    for sample in samples:
        by_type = {}
        for event in sample["events"]:
            by_type.setdefault(event["event_type"], []).extend(event["trigger"])
        for event_type, spans in by_type.items():
            examples += _answered(sample, trigger_question(event_type, templates), spans)
    return _unique_examples(examples)


def _match_event_by_trigger(sample, trigger):
    candidates = [
        (min(_distance(trigger["start"], trigger["end"], s["start"], s["end"]) for s in event["trigger"]), index)
        for index, event in enumerate(sample["events"])
        if event["event_type"] == trigger["event_type"] and event["trigger"]
    ]
    return sample["events"][min(candidates)[1]] if candidates else None


def main_training_examples(samples, templates, predicted_triggers=None):
    examples = []
    for index, sample in enumerate(samples):
        if predicted_triggers is None:
            pairs = [(event["trigger"][0], event["event_type"], event) for event in sample["events"] if event["trigger"]]
        else:
            pairs = [
                (trigger, trigger["event_type"], _match_event_by_trigger(sample, trigger))
                for trigger in predicted_triggers[index]
            ]
        for trigger, event_type, event in pairs:
            for label in MAIN_LABELS:
                question = main_question(label, trigger["text"], event_type, templates)
                spans = event["args"][label] if event else []
                examples += _answered(sample, question, spans)
    return _unique_examples(examples)


def _match_event_by_main(sample, main):
    candidates = [
        (min(_distance(main["start"], main["end"], s["start"], s["end"]) for s in event["args"][main["label"]]), index)
        for index, event in enumerate(sample["events"])
        if event["args"].get(main["label"])
    ]
    return sample["events"][min(candidates)[1]] if candidates else None


def sub_training_examples(samples, templates, predicted_mains=None):
    examples = []
    for index, sample in enumerate(samples):
        if predicted_mains is None:
            pairs = [
                (span, parent, event["event_type"], event)
                for event in sample["events"]
                for parent in SUB_LABELS_BY_PARENT
                for span in event["args"][parent]
            ]
        else:
            pairs = [
                (main, main["label"], main.get("event_type"), _match_event_by_main(sample, main))
                for main in predicted_mains[index]
                if main["label"] in SUB_LABELS_BY_PARENT
            ]
        for main, parent, event_type, event in pairs:
            for label in SUB_LABELS_BY_PARENT[parent]:
                question = sub_question(label, main["text"], event_type, templates)
                spans = event["subargs"][label] if event else []
                examples += _answered(sample, question, spans)
    return _unique_examples(examples)


# ------------------------------------------------------------------- prediction steps

def _ask(model, tokenizer, queries, settings, answer_settings, model_config):
    """queries: list of (sample_index, context, question, meta).
    Identical questions of one sentence are asked once.
    Returns a list of (sample_index, meta, answers), one per query."""
    unique_keys = []
    position = {}
    for sample_index, context, question, _ in queries:
        key = (sample_index, question)
        if key not in position:
            position[key] = len(unique_keys)
            unique_keys.append((key, context))
    examples = [{"question": key[1], "context": context} for key, context in unique_keys]
    answers = predict_answers(
        model, tokenizer, examples, settings,
        max_answer_length=answer_settings["max_answer_length"],
        max_spans=answer_settings["max_spans"],
        use_null=answer_settings["use_null"],
        null_threshold=answer_settings["null_threshold"],
        max_length=model_config.get("max_length", 384),
        stride=model_config.get("stride", 128),
    )
    return [
        (sample_index, meta, answers[position[(sample_index, question)]])
        for sample_index, _, question, meta in queries
    ]


def predict_triggers(model, tokenizer, samples, event_types, templates, settings, answer_settings, model_config):
    """event_types: list (per sample) of event types to ask about."""
    queries = [
        (index, sample["context"], trigger_question(event_type, templates), event_type)
        for index, sample in enumerate(samples)
        for event_type in event_types[index]
    ]
    triggers = [[] for _ in samples]
    for index, event_type, answers in _ask(model, tokenizer, queries, settings, answer_settings, model_config):
        for answer in answers:
            triggers[index].append({
                "label": "TRIGGER",
                "event_type": event_type,
                "text": answer["text"],
                "start": answer["start"],
                "end": answer["end"],
                "score": answer["score"],
            })
    return triggers


def predict_mains(model, tokenizer, samples, triggers, templates, settings, answer_settings, model_config):
    queries = [
        (index, sample["context"], main_question(label, trigger["text"], trigger["event_type"], templates), (label, trigger))
        for index, sample in enumerate(samples)
        for trigger in triggers[index]
        for label in MAIN_LABELS
    ]
    mains = [[] for _ in samples]
    for index, (label, trigger), answers in _ask(model, tokenizer, queries, settings, answer_settings, model_config):
        for answer in answers:
            mains[index].append({
                "label": label,
                "text": answer["text"],
                "start": answer["start"],
                "end": answer["end"],
                "score": answer["score"],
                "event_type": trigger["event_type"],
                "trigger_text": trigger["text"],
            })
    return mains


def predict_subs(model, tokenizer, samples, mains, templates, settings, answer_settings, model_config):
    queries = [
        (index, sample["context"], sub_question(label, main["text"], main.get("event_type"), templates), (label, main))
        for index, sample in enumerate(samples)
        for main in mains[index]
        if main["label"] in SUB_LABELS_BY_PARENT
        for label in SUB_LABELS_BY_PARENT[main["label"]]
    ]
    subs = [[] for _ in samples]
    for index, (label, main), answers in _ask(model, tokenizer, queries, settings, answer_settings, model_config):
        for answer in answers:
            subs[index].append({
                "label": label,
                "text": answer["text"],
                "start": answer["start"],
                "end": answer["end"],
                "score": answer["score"],
                "main_text": main["text"],
                "event_type": main.get("event_type"),
            })
    return subs


def gold_triggers(samples):
    return [
        [
            {"label": "TRIGGER", "event_type": event["event_type"], "text": span["text"], "start": span["start"], "end": span["end"]}
            for event in sample["events"]
            for span in event["trigger"]
        ]
        for sample in samples
    ]


def gold_mains(samples):
    return [
        [
            {"label": label, "text": span["text"], "start": span["start"], "end": span["end"], "event_type": event["event_type"]}
            for event in sample["events"]
            for label, spans in event["args"].items()
            for span in spans
        ]
        for sample in samples
    ]


def _records(samples, *item_lists):
    records = []
    for index, sample in enumerate(samples):
        pred = []
        for items in item_lists:
            for item in items[index]:
                prediction = {"label": item["label"], "text": item["text"]}
                if item["label"] == "TRIGGER":
                    prediction["event_type"] = item["event_type"]
                pred.append(prediction)
        records.append({"id": sample["id"], "pred": pred})
    return records


# ------------------------------------------------------------------------- experiment

def _stage_settings(config, stage):
    base = training_settings(config.get("training"))
    return deep_update(base, (config.get("stage_training") or {}).get(stage, {}))


def run(config, seed, run_dir):
    from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification

    data = load_splits(config)
    model_config = config["model"]
    templates = config.get("templates", "current")
    train_inputs = config.get("train_inputs", "predicted")
    answer_config = deep_update(DEFAULT_ANSWERS, config.get("answers") or {})
    event_config = {"threshold": 0.5, "force_one": True, "max_length": 256}
    event_config.update(config.get("event_type") or {})
    recorder = RunRecorder(run_dir, config, seed)
    es_dev = early_stopping_dev(config, data)
    tokenizer = load_tokenizer(model_config["name"])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    load_kwargs = {"attn_implementation": model_config["attn_implementation"]} if model_config.get("attn_implementation") else {}
    need_train_predictions = train_inputs == "predicted"
    splits = ["dev", "test"] + (["train"] if need_train_predictions else [])
    inference = {split: {"seconds": 0.0, "peak_gpu_memory_gb": None} for split in ("dev", "test")}

    def track(split, timer, memory):
        if split in inference:
            inference[split]["seconds"] += timer.seconds
            peaks = [p for p in (inference[split]["peak_gpu_memory_gb"], memory.peak_gb) if p is not None]
            inference[split]["peak_gpu_memory_gb"] = max(peaks) if peaks else None

    # 1. event type classifier ---------------------------------------------------------
    settings = _stage_settings(config, "event_type")
    log(f"eqa: stage 1 event type classifier ({model_config['name']})")
    set_seed(seed)  # new heads are initialised with this seed
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["name"], num_labels=2, problem_type="multi_label_classification", **load_kwargs
    )
    train_items = event_type_stage.features(data["train"], tokenizer, event_config["max_length"])
    model, report = train_model(
        model, train_items, event_type_stage.make_collate(pad_id),
        lambda m: event_type_stage.score(
            es_dev,
            event_type_stage.predict(m, tokenizer, es_dev, settings, event_config["threshold"],
                                     event_config["force_one"], event_config["max_length"]),
        ),
        settings, seed, name="eqa_event_type",
    )
    report["parameters"] = count_parameters(model)
    recorder.add_stage(report)
    event_types = {}
    for split in splits:
        with GpuMemory() as memory, Timer() as timer:
            predictions = event_type_stage.predict(
                model, tokenizer, data[split], settings, event_config["threshold"],
                event_config["force_one"], event_config["max_length"],
            )
        track(split, timer, memory)
        event_types[split] = [p["types"] for p in predictions]
    del model
    cleanup()

    # 2. trigger QA -----------------------------------------------------------------
    settings = _stage_settings(config, "trigger")
    log("eqa: stage 2 trigger QA")
    set_seed(seed)  # new heads are initialised with this seed
    model = AutoModelForQuestionAnswering.from_pretrained(model_config["name"], **load_kwargs)
    use_types = accepts_token_type_ids(model)
    qa_collate = make_collate(pad_id)
    max_length, stride = model_config.get("max_length", 384), model_config.get("stride", 128)
    train_items = train_features(trigger_training_examples(data["train"], templates), tokenizer, max_length, stride, use_types)
    dev_gold_types = [sorted({event["event_type"] for event in sample["events"]}) for sample in es_dev]
    model, report = train_model(
        model, train_items, qa_collate,
        lambda m: score_families(
            es_dev,
            _records(es_dev, predict_triggers(m, tokenizer, es_dev, dev_gold_types, templates,
                                                   settings, answer_config["trigger"], model_config)),
            ["trigger_classification"],
        ),
        settings, seed, name="eqa_trigger",
    )
    report["parameters"] = count_parameters(model)
    recorder.add_stage(report)
    triggers = {}
    for split in splits:
        with GpuMemory() as memory, Timer() as timer:
            triggers[split] = predict_triggers(model, tokenizer, data[split], event_types[split], templates,
                                               settings, answer_config["trigger"], model_config)
        track(split, timer, memory)
    del model
    cleanup()

    # 3. main-argument QA -----------------------------------------------------------
    settings = _stage_settings(config, "main")
    log(f"eqa: stage 3 main-argument QA (train inputs: {train_inputs})")
    set_seed(seed)  # new heads are initialised with this seed
    model = AutoModelForQuestionAnswering.from_pretrained(model_config["name"], **load_kwargs)
    examples = main_training_examples(
        data["train"], templates, triggers["train"] if need_train_predictions else None
    )
    main_fallback = not examples
    if main_fallback:
        log("eqa: WARNING no main-argument training examples from predictions, using gold triggers")
        examples = main_training_examples(data["train"], templates)
    train_items = train_features(examples, tokenizer, max_length, stride, use_types)
    dev_gold_triggers = gold_triggers(es_dev)
    model, report = train_model(
        model, train_items, qa_collate,
        lambda m: score_families(
            es_dev,
            _records(es_dev, predict_mains(m, tokenizer, es_dev, dev_gold_triggers, templates,
                                                settings, answer_config["main"], model_config)),
            ["main_classification"],
        ),
        settings, seed, name="eqa_main",
    )
    report["parameters"] = count_parameters(model)
    report["fallback_to_gold_inputs"] = main_fallback
    recorder.add_stage(report)
    mains = {}
    for split in splits:
        with GpuMemory() as memory, Timer() as timer:
            mains[split] = predict_mains(model, tokenizer, data[split], triggers[split], templates,
                                         settings, answer_config["main"], model_config)
        track(split, timer, memory)
    del model
    cleanup()

    # 4. sub-argument QA ------------------------------------------------------------
    settings = _stage_settings(config, "sub")
    log(f"eqa: stage 4 sub-argument QA (train inputs: {train_inputs})")
    set_seed(seed)  # new heads are initialised with this seed
    model = AutoModelForQuestionAnswering.from_pretrained(model_config["name"], **load_kwargs)
    examples = sub_training_examples(
        data["train"], templates, mains["train"] if need_train_predictions else None
    )
    sub_fallback = not examples
    if sub_fallback:
        log("eqa: WARNING no sub-argument training examples from predictions, using gold main arguments")
        examples = sub_training_examples(data["train"], templates)
    train_items = train_features(examples, tokenizer, max_length, stride, use_types)
    dev_gold_mains = gold_mains(es_dev)
    model, report = train_model(
        model, train_items, qa_collate,
        lambda m: score_families(
            es_dev,
            _records(es_dev, predict_subs(m, tokenizer, es_dev, dev_gold_mains, templates,
                                               settings, answer_config["sub"], model_config)),
            ["sub_classification"],
        ),
        settings, seed, name="eqa_sub",
    )
    report["parameters"] = count_parameters(model)
    report["fallback_to_gold_inputs"] = sub_fallback
    recorder.add_stage(report)
    subs = {}
    for split in ("dev", "test"):
        with GpuMemory() as memory, Timer() as timer:
            subs[split] = predict_subs(model, tokenizer, data[split], mains[split], templates,
                                       settings, answer_config["sub"], model_config)
        track(split, timer, memory)
    del model
    cleanup()

    variant = f"templates={templates},train_inputs={train_inputs}"
    for split in ("dev", "test"):
        records = _records(data[split], triggers[split], mains[split], subs[split])
        for record, trig, main, sub in zip(records, triggers[split], mains[split], subs[split]):
            record["structure"] = {"triggers": trig, "mains": main, "subs": sub}
        recorder.add_variant(variant, split, data[split], records, inference[split])
    return recorder.finish()

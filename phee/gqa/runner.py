"""Generative QA pipeline experiment (two stages, as in the PHEE paper).

  stage 1  "question: What are the events? context: <sentence>"
           -> events with trigger and main arguments (phee.gqa.format)
  stage 2  one question per (predicted main argument, sub-argument label)
           -> "answer ; answer" or "none"

Variants:
  target_format  own | authors                  (stage-1 target, see phee.gqa.format)
  templates      authors | current | definitions (stage-2 questions, phee.eqa.templates)
  train_inputs   gold | predicted               (stage-2 training questions from gold main
                                                 arguments or from stage-1 predictions on train)
  decodings      [free, constrained]            (evaluated on the same trained models)

Evaluation uses the RAW generated text (no snapping to the sentence).
Early stopping: after every epoch the dev set is generated (beam search, num_beams) and
scored with EM-F1 (stage 1: triggers + main arguments, stage 2: sub-arguments from
gold main arguments), as in the authors' configuration.
"""

import torch

from phee.eqa.templates import sub_question
from phee.evaluation import score_families
from phee.experiment import RunRecorder, early_stopping_dev, load_splits
from phee.gqa.constrained import ConstraintBuilder
from phee.gqa.format import (
    EVENT_MARKERS,
    NONE_TEXT,
    ROLE_MARKERS,
    SEP_TEXT,
    STAGE1_QUESTION,
    SegmentTokenizer,
    build_input,
    parse_stage1,
    parse_stage2,
    stage1_segments,
    stage2_segments,
)
from phee.hf import count_parameters, load_tokenizer
from phee.io import deep_update
from phee.normalize import normalize_text
from phee.schema import SUB_LABELS_BY_PARENT
from phee.training import train_model, training_settings
from phee.utils import GpuMemory, Timer, autocast_context, cleanup, get_device, log, set_seed

DEFAULT_GENERATION = {
    "num_beams": 3,
    "stage1_max_new_tokens": 256,
    "stage2_max_new_tokens": 64,
    "max_source_length": 512,
    "max_target_length": 256,
    "max_span_words": 80,
}


def _collate(pad_token_id):
    def collate(items):
        length = max(len(item["input_ids"]) for item in items)
        input_ids = torch.full((len(items), length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(items), length), dtype=torch.long)
        for row, item in enumerate(items):
            n = len(item["input_ids"])
            input_ids[row, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[row, :n] = 1
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "labels" in items[0]:
            target_length = max(len(item["labels"]) for item in items)
            labels = torch.full((len(items), target_length), -100, dtype=torch.long)
            for row, item in enumerate(items):
                labels[row, :len(item["labels"])] = torch.tensor(item["labels"], dtype=torch.long)
            batch["labels"] = labels
        return batch

    return collate


class Generator:
    """Batched generation with optional constraints."""

    def __init__(self, tokenizer, generation, settings):
        self.tokenizer = tokenizer
        self.generation = generation
        self.settings = settings

    def __call__(self, model, inputs, samples, max_new_tokens, constraint=None):
        """inputs: list of source strings; samples: sentence of every input (constraints)."""
        device = get_device()
        outputs = []
        batch_size = self.settings.get("eval_batch_size", 32)
        model.eval()
        for start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[start:start + batch_size]
            encoded = self.tokenizer(
                batch_inputs,
                max_length=self.generation["max_source_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            kwargs = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "num_beams": self.generation["num_beams"],
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
            if constraint is not None:
                kwargs["prefix_allowed_tokens_fn"] = constraint.prefix_allowed_tokens_fn(
                    samples[start:start + batch_size]
                )
            with torch.no_grad(), autocast_context(self.settings.get("precision", "bf16")):
                generated = model.generate(**kwargs)
            outputs.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        if constraint is not None:
            constraint.clear()
        return outputs


def _stage1_items(samples, tokenizer, segment_tokenizer, target_format, generation):
    items = []
    for sample in samples:
        source = tokenizer(
            build_input(STAGE1_QUESTION, sample["context"]),
            max_length=generation["max_source_length"],
            truncation=True,
        )["input_ids"]
        labels = segment_tokenizer.encode(stage1_segments(sample, target_format), generation["max_target_length"])
        items.append({"input_ids": source, "labels": labels})
    return items


def _stage1_predictions(texts, split_values):
    """Parsed stage-1 outputs -> (triggers, mains) per sample in the unified format."""
    triggers, mains = [], []
    for text in texts:
        sample_triggers, sample_mains = [], []
        for event in parse_stage1(text, split_values=split_values):
            for trigger in event["trigger"]:
                sample_triggers.append({"label": "TRIGGER", "event_type": event["event_type"], "text": trigger})
            for label, values in event["args"].items():
                for value in values:
                    sample_mains.append({"label": label, "text": value, "event_type": event["event_type"]})
        triggers.append(sample_triggers)
        mains.append(sample_mains)
    return triggers, mains


def _token_f1(a, b):
    a_tokens, b_tokens = normalize_text(a).split(), normalize_text(b).split()
    if not a_tokens or not b_tokens:
        return 0.0
    common = sum(min(a_tokens.count(t), b_tokens.count(t)) for t in set(a_tokens))
    if not common:
        return 0.0
    precision, recall = common / len(b_tokens), common / len(a_tokens)
    return 2 * precision * recall / (precision + recall)


def _match_event(sample, main):
    """Gold event whose main argument of the same label is most similar to the
    predicted text (generated text has no position)."""
    best, best_score = None, 0.0
    for event in sample["events"]:
        for span in event["args"].get(main["label"], []):
            score = _token_f1(span["text"], main["text"])
            if score > best_score:
                best, best_score = event, score
    return best


def _stage2_queries(samples, mains, templates):
    """One query per (sample, main argument, sub label), identical questions of one
    sentence merged. Returns list of (sample_index, question, label)."""
    queries = []
    seen = set()
    for index, sample in enumerate(samples):
        for main in mains[index]:
            for label in SUB_LABELS_BY_PARENT.get(main["label"], []):
                question = sub_question(label, main["text"], main.get("event_type"), templates)
                key = (index, question)
                if key in seen:
                    continue
                seen.add(key)
                queries.append((index, question, label))
    return queries


def _stage2_training_items(samples, tokenizer, segment_tokenizer, templates, generation, predicted_mains=None):
    items = []
    seen = set()
    for index, sample in enumerate(samples):
        if predicted_mains is None:
            pairs = [
                ({"label": parent, "text": span["text"], "event_type": event["event_type"]}, event)
                for event in sample["events"]
                for parent in SUB_LABELS_BY_PARENT
                for span in event["args"][parent]
            ]
        else:
            pairs = [(main, _match_event(sample, main)) for main in predicted_mains[index]
                     if main["label"] in SUB_LABELS_BY_PARENT]
        for main, event in pairs:
            for label in SUB_LABELS_BY_PARENT[main["label"]]:
                question = sub_question(label, main["text"], main.get("event_type"), templates)
                answers = []
                if event is not None:
                    for span in event["subargs"][label]:
                        if span["text"] not in answers:
                            answers.append(span["text"])
                key = (index, question, tuple(answers))
                if key in seen:
                    continue
                seen.add(key)
                source = tokenizer(
                    build_input(question, sample["context"]),
                    max_length=generation["max_source_length"],
                    truncation=True,
                )["input_ids"]
                labels = segment_tokenizer.encode(stage2_segments(answers), generation["max_target_length"])
                items.append({"input_ids": source, "labels": labels})
    return items


def _predict_stage2(model, generate, samples, mains, templates, generation, constraint):
    queries = _stage2_queries(samples, mains, templates)
    inputs = [build_input(question, samples[index]["context"]) for index, question, _ in queries]
    texts = generate(model, inputs, [samples[index] for index, _, _ in queries],
                     generation["stage2_max_new_tokens"], constraint)
    subs = [[] for _ in samples]
    for (index, _, label), text in zip(queries, texts):
        for value in parse_stage2(text):
            subs[index].append({"label": label, "text": value})
    return subs


def _records(samples, *lists):
    records = []
    for index, sample in enumerate(samples):
        pred = []
        for items in lists:
            for item in items[index]:
                prediction = {"label": item["label"], "text": item["text"]}
                if item["label"] == "TRIGGER":
                    prediction["event_type"] = item["event_type"]
                pred.append(prediction)
        records.append({"id": sample["id"], "pred": pred})
    return records


def _gold_mains(samples):
    return [
        [
            {"label": label, "text": span["text"], "event_type": event["event_type"]}
            for event in sample["events"]
            for label, spans in event["args"].items()
            for span in spans
        ]
        for sample in samples
    ]


def _stage_settings(config, stage):
    base = training_settings(config.get("training"))
    return deep_update(base, (config.get("stage_training") or {}).get(stage, {}))


def run(config, seed, run_dir):
    from transformers import AutoModelForSeq2SeqLM

    data = load_splits(config)
    model_config = config["model"]
    target_format = config.get("target_format", "own")
    templates = config.get("templates", "current")
    train_inputs = config.get("train_inputs", "predicted")
    decodings = config.get("decodings", ["free", "constrained"])
    generation = deep_update(DEFAULT_GENERATION, config.get("generation") or {})
    split_values = target_format == "authors"
    recorder = RunRecorder(run_dir, config, seed)
    es_dev = early_stopping_dev(config, data)

    tokenizer = load_tokenizer(model_config["name"])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    segment_tokenizer = SegmentTokenizer(tokenizer)
    load_kwargs = {"attn_implementation": model_config["attn_implementation"]} if model_config.get("attn_implementation") else {}
    stage1_grammar = "stage1_authors" if target_format == "authors" else "stage1_own"
    symbols1 = {
        "event": list(EVENT_MARKERS[target_format].values()),
        "role": list(ROLE_MARKERS[target_format].values()),
        "sep": [SEP_TEXT],
        "none": [NONE_TEXT],
    }
    symbols2 = {"sep": [SEP_TEXT], "none": [NONE_TEXT]}
    constraints = {
        "free": (None, None),
        "constrained": (
            ConstraintBuilder(segment_tokenizer, symbols1, stage1_grammar, generation["max_span_words"]),
            ConstraintBuilder(segment_tokenizer, symbols2, "stage2", generation["max_span_words"]),
        ),
    }
    need_train_predictions = train_inputs == "predicted"
    inference = {(d, s): {"seconds": 0.0, "peak_gpu_memory_gb": None} for d in decodings for s in ("dev", "test")}

    def track(key, timer, memory):
        inference[key]["seconds"] += timer.seconds
        peaks = [p for p in (inference[key]["peak_gpu_memory_gb"], memory.peak_gb) if p is not None]
        inference[key]["peak_gpu_memory_gb"] = max(peaks) if peaks else None

    # stage 1 ---------------------------------------------------------------------
    settings = _stage_settings(config, "stage1")
    generate = Generator(tokenizer, generation, settings)
    log(f"gqa: stage 1 ({model_config['name']}, target format {target_format})")
    set_seed(seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_config["name"], **load_kwargs)
    train_items = _stage1_items(data["train"], tokenizer, segment_tokenizer, target_format, generation)
    dev_inputs = [build_input(STAGE1_QUESTION, sample["context"]) for sample in es_dev]

    def evaluate_stage1(current_model):
        texts = generate(current_model, dev_inputs, es_dev, generation["stage1_max_new_tokens"])
        triggers, mains = _stage1_predictions(texts, split_values)
        return score_families(es_dev, _records(es_dev, triggers, mains),
                              ["trigger_classification", "main_classification"])

    model, report = train_model(model, train_items, _collate(pad_id), evaluate_stage1, settings, seed,
                                name="gqa_stage1")
    report["parameters"] = count_parameters(model)
    recorder.add_stage(report)

    stage1 = {}
    raw1 = {}
    for decoding in decodings:
        for split in ("dev", "test"):
            inputs = [build_input(STAGE1_QUESTION, sample["context"]) for sample in data[split]]
            with GpuMemory() as memory, Timer() as timer:
                texts = generate(model, inputs, data[split], generation["stage1_max_new_tokens"],
                                 constraints[decoding][0])
            track((decoding, split), timer, memory)
            raw1[(decoding, split)] = texts
            stage1[(decoding, split)] = _stage1_predictions(texts, split_values)
    if need_train_predictions:
        inputs = [build_input(STAGE1_QUESTION, sample["context"]) for sample in data["train"]]
        with Timer() as timer:
            texts = generate(model, inputs, data["train"], generation["stage1_max_new_tokens"])
        train_mains = _stage1_predictions(texts, split_values)[1]
        recorder.stages[-1]["train_set_prediction_seconds"] = timer.seconds
    del model
    cleanup()

    # stage 2 ---------------------------------------------------------------------
    settings = _stage_settings(config, "stage2")
    generate = Generator(tokenizer, generation, settings)
    log(f"gqa: stage 2 (templates {templates}, train inputs {train_inputs})")
    set_seed(seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_config["name"], **load_kwargs)
    train_items = _stage2_training_items(
        data["train"], tokenizer, segment_tokenizer, templates, generation,
        train_mains if need_train_predictions else None,
    )
    fallback = False
    if not train_items:
        # stage 1 predicted no main argument on the training set (e.g. a broken model):
        # train stage 2 on gold inputs instead of failing, and record it
        log("gqa: WARNING no stage-2 training examples from predictions, using gold main arguments")
        train_items = _stage2_training_items(data["train"], tokenizer, segment_tokenizer, templates, generation)
        fallback = True
    dev_gold_mains = _gold_mains(es_dev)

    def evaluate_stage2(current_model):
        subs = _predict_stage2(current_model, generate, es_dev, dev_gold_mains, templates, generation, None)
        return score_families(es_dev, _records(es_dev, subs), ["sub_classification"])

    model, report = train_model(model, train_items, _collate(pad_id), evaluate_stage2, settings, seed,
                                name="gqa_stage2")
    report["parameters"] = count_parameters(model)
    report["fallback_to_gold_inputs"] = fallback
    recorder.add_stage(report)

    for decoding in decodings:
        for split in ("dev", "test"):
            triggers, mains = stage1[(decoding, split)]
            with GpuMemory() as memory, Timer() as timer:
                subs = _predict_stage2(model, generate, data[split], mains, templates, generation,
                                       constraints[decoding][1])
            track((decoding, split), timer, memory)
            records = _records(data[split], triggers, mains, subs)
            for record, text in zip(records, raw1[(decoding, split)]):
                record["generated_stage1"] = text
            variant = f"format={target_format},templates={templates},train_inputs={train_inputs},decoding={decoding}"
            recorder.add_variant(variant, split, data[split], records, inference[(decoding, split)])
    del model
    cleanup()
    return recorder.finish()

"""Sequence labeling experiment: train, predict with every decoding, evaluate.

Config (configs/seq/*.yaml):
  method: seq
  scheme: two_level | per_trigger | authors_flat
    two_level: one model for triggers + main arguments, one for sub-arguments; events
      rebuilt by the nearest-trigger rule (phee.events.assign_by_nearest_trigger)
    per_trigger: a trigger tagger, then main-argument and sub-argument taggers run once
      per predicted trigger on the sentence with that trigger marked [TRG] ... [/TRG]
      (MAILEX, Srivastava et al. 2023); everything tagged belongs to that trigger's event
      train_inputs: gold | predicted   argument taggers trained on the gold triggers, or
        on the triggers the trigger tagger predicts on the training sentences (each
        matched to the nearest gold event of its type, as in the extractive QA; no match:
        an example without arguments); early stopping then also uses predicted triggers
      min_triggers: 0 | 1   with 1, a sentence where the trigger tagger finds nothing gets
        the word with the highest trigger score (every PHEE sentence has an event)
  model:
    name: thomas-sounack/BioClinical-ModernBERT-base
    head: linear | mlp | crf
    dropout, mlp_hidden_size, mlp_dropout, mlp_activation, crf_constrained, max_length
  training: see phee.training.DEFAULT_TRAINING (+ crf_learning_rate)
  decodings: [argmax, viterbi]
"""

from collections import defaultdict

import torch

from phee.events import assign_by_nearest_trigger
from phee.evaluation import score_families
from phee.experiment import RunRecorder, early_stopping_dev, load_splits
from phee.schema import TRIGGER
from phee.hf import count_parameters, load_encoder, load_tokenizer, model_kwargs
from phee.seq.decoding import decode, native_decoding
from phee.seq.encoding import WordEncoder, make_collate
from phee.seq.model import TokenTagger
from phee.seq.tagging import (
    ARGUMENT_MAIN_TAGS,
    AUTHORS_TAGS,
    TRIGGER_MARKERS,
    TRIGGER_TAGS,
    OWN_MAIN_TAGS,
    OWN_SUB_TAGS,
    authors_tags_to_entities,
    authors_training_sequences,
    bio_to_entities,
    entity_to_prediction,
    event_main_tags,
    event_sub_tags,
    own_main_tags,
    own_sub_tags,
    trigger_groups,
    trigger_tags,
)
from phee.training import train_model, training_settings
from phee.utils import GpuMemory, Timer, autocast_context, cleanup, get_device, log, set_seed


def _levels(scheme):
    if scheme == "two_level":
        return [
            {
                "name": "main",
                "tags": OWN_MAIN_TAGS,
                "gold": lambda sample: [own_main_tags(sample)[0]],
                "entities": bio_to_entities,
                "families": ["trigger_classification", "main_classification"],
            },
            {
                "name": "sub",
                "tags": OWN_SUB_TAGS,
                "gold": lambda sample: [own_sub_tags(sample)[0]],
                "entities": bio_to_entities,
                "families": ["sub_classification"],
            },
        ]
    if scheme == "per_trigger":
        return [
            {
                "name": "trigger",
                "tags": TRIGGER_TAGS,
                "gold": lambda sample: [trigger_tags(sample)[0]],
                "entities": bio_to_entities,
                "families": ["trigger_classification"],
            },
            {
                "name": "main",
                "tags": ARGUMENT_MAIN_TAGS,
                "per_trigger": True,
                "event_gold": lambda sample, events: event_main_tags(sample, events)[0],
                "entities": bio_to_entities,
                "families": ["main_classification"],
            },
            {
                "name": "sub",
                "tags": OWN_SUB_TAGS,
                "per_trigger": True,
                "event_gold": lambda sample, events: event_sub_tags(sample, events)[0],
                "entities": bio_to_entities,
                "families": ["sub_classification"],
            },
        ]
    if scheme == "authors_flat":
        return [
            {
                "name": "flat",
                "tags": AUTHORS_TAGS,
                "gold": authors_training_sequences,   # one copy of the sentence per event
                "entities": authors_tags_to_entities,
                "families": ["trigger_classification", "main_classification", "sub_classification"],
            }
        ]
    raise ValueError(f"Unknown scheme: {scheme}")


def _build_items(samples, encoder, level, with_labels):
    tag_index = {tag: i for i, tag in enumerate(level["tags"])}
    items = []
    for sample in samples:
        encoded = encoder.encode(sample)
        if not with_labels:
            items.append(encoded)
            continue
        for tags in level["gold"](sample):
            item = dict(encoded)
            item["labels"] = [tag_index[tags[word]] for word in encoded["valid_words"]]
            items.append(item)
    return items


def _marked_training_items(samples, encoder, level):
    """One item per gold trigger: the sentence with that trigger marked, tags of the
    arguments of its event only."""
    tag_index = {tag: i for i, tag in enumerate(level["tags"])}
    items = []
    for sample in samples:
        for marks, events in trigger_groups(sample):
            item = encoder.encode(sample, marks)
            tags = level["event_gold"](sample, events)
            item["labels"] = [tag_index[tags[word]] for word in item["valid_words"]]
            items.append(item)
    return items


def _gold_triggers(samples):
    """Per sample: [(marks, event index, event type)] from the gold events (dev early stopping)."""
    return [
        [(marks, index, events[0]["event_type"]) for index, (marks, events) in enumerate(trigger_groups(sample))]
        for sample in samples
    ]


def _predicted_triggers(preds):
    """Trigger predictions of one sample -> [(marks, event index, event type)]; every
    predicted trigger is one event (its index is stored on the trigger item)."""
    triggers = []
    for item in preds:
        if item["label"] == TRIGGER and "tok_start" in item:
            item["event"] = len(triggers)
            triggers.append(([(item["tok_start"], item["tok_end"])], item["event"], item["event_type"]))
    return triggers


def _marked_queries(samples, triggers, encoder):
    """Items (one per trigger) and their owners (sample index, event index, event type)."""
    items, owners = [], []
    for sample_index, (sample, sample_triggers) in enumerate(zip(samples, triggers)):
        for marks, event, event_type in sample_triggers:
            items.append(encoder.encode(sample, marks))
            owners.append((sample_index, event, event_type))
    return items, owners


def _marked_predictions(model, samples, items, owners, emissions, level, strategy):
    """Decoded predictions of the per-trigger items, attached to their events and
    grouped per sample."""
    aligned = [samples[sample_index] for sample_index, _, _ in owners]
    grouped = [[] for _ in samples]
    per_item = _decode_predictions(model, aligned, items, emissions, level, strategy)
    for (sample_index, event, event_type), preds in zip(owners, per_item):
        for prediction in preds:
            prediction["event"] = event
            prediction["event_type"] = event_type
            grouped[sample_index].append(prediction)
    return grouped


def _add_inference(record, seconds, peak_gb):
    record["seconds"] += seconds
    record["peak_gpu_memory_gb"] = max(
        [value for value in (record["peak_gpu_memory_gb"], peak_gb) if value is not None], default=None
    )


def _emissions(model, items, collate, batch_size, precision):
    """Word-level emissions for every item (kept on CPU)."""
    device = get_device()
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = collate(items[start:start + batch_size])
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(precision):
                result = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    word_positions=batch["word_positions"],
                    word_mask=batch["word_mask"],
                )
            outputs.append((result.emissions.float(), batch["word_mask"]))
    return outputs


def forced_trigger(sample, item, scores, tag_names):
    """The single-word trigger with the highest B-*_TRIGGER emission (min_triggers: 1)."""
    best = None
    for tag_id, tag in enumerate(tag_names):
        if not (tag.startswith("B-") and tag.endswith("_TRIGGER")):
            continue
        for position, word in enumerate(item["valid_words"]):
            score = float(scores[position, tag_id])
            if best is None or score > best[0]:
                best = (score, word, tag[2:])
    if best is None:
        return None
    return entity_to_prediction(sample, best[1], best[1], best[2])


def _decode_predictions(model, samples, items, emissions, level, strategy, min_triggers=0):
    """Predictions (unified format) of one level for every sample."""
    predictions = []
    index = 0
    for scores, word_mask in emissions:
        for row_index, row in enumerate(decode(model, scores, word_mask, strategy)):
            sample = samples[index]
            item = items[index]
            tags = ["O"] * len(sample["tokens"])
            for position, tag_id in enumerate(row):
                tags[item["valid_words"][position]] = model.tag_names[tag_id]
            preds = []
            for start, end, entity in level["entities"](tags):
                prediction = entity_to_prediction(sample, start, end, entity)
                if prediction:
                    preds.append(prediction)
            if min_triggers and not any(p["label"] == TRIGGER for p in preds) and item["valid_words"]:
                prediction = forced_trigger(sample, item, scores[row_index], model.tag_names)
                if prediction:
                    preds.append(prediction)
            predictions.append(preds)
            index += 1
    return predictions


def match_events(sample, marks, event_type):
    """Gold events of `event_type` nearest (in tokens) to a predicted trigger; several
    when they share that trigger. Same matching as the extractive QA predicted inputs."""
    start, end = marks[0]
    candidates = []
    for event in sample["events"]:
        if event["event_type"] != event_type or not event["trigger"]:
            continue
        distance = min(
            max(0, span["tok_start"] - end, start - span["tok_end"]) for span in event["trigger"]
        )
        candidates.append((distance, event))
    if not candidates:
        return []
    nearest = min(distance for distance, _ in candidates)
    return [event for distance, event in candidates if distance == nearest]


def _predicted_training_items(samples, triggers, encoder, level):
    """One item per trigger predicted on a training sentence, with the tags of the matched
    gold event(s); a trigger without a matching event gives an item without arguments."""
    tag_index = {tag: i for i, tag in enumerate(level["tags"])}
    items = []
    for sample, sample_triggers in zip(samples, triggers):
        for marks, _, event_type in sample_triggers:
            item = encoder.encode(sample, marks)
            events = match_events(sample, marks, event_type)
            tags = level["event_gold"](sample, events) if events else ["O"] * len(sample["tokens"])
            item["labels"] = [tag_index[tags[word]] for word in item["valid_words"]]
            items.append(item)
    return items


def run(config, seed, run_dir):
    data = load_splits(config)
    model_config = config["model"]
    settings = training_settings(config.get("training"))
    decodings = config.get("decodings", ["argmax"])
    levels = _levels(config["scheme"])
    recorder = RunRecorder(run_dir, config, seed)
    es_dev = early_stopping_dev(config, data)

    per_trigger = config["scheme"] == "per_trigger"
    train_inputs = config.get("train_inputs", "gold")
    min_triggers = int(config.get("min_triggers", 0)) if per_trigger else 0
    stage1 = {}     # per_trigger + train_inputs predicted: triggers predicted on train / es_dev
    tokenizer = load_tokenizer(model_config["name"])
    encoder_helper = WordEncoder(tokenizer, max_length=model_config.get("max_length", 512))
    marked_helper = None
    if per_trigger:
        tokenizer.add_tokens(list(TRIGGER_MARKERS), special_tokens=True)
        marked_helper = WordEncoder(tokenizer, max_length=model_config.get("max_length", 512), markers=TRIGGER_MARKERS)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = make_collate(pad_id)

    # predictions[split][decoding] -> list (per sample) of prediction lists, merged over levels
    merged = {split: {d: [[] for _ in data[split]] for d in decodings} for split in ("dev", "test")}
    inference = {split: defaultdict(lambda: {"seconds": 0.0, "peak_gpu_memory_gb": None}) for split in ("dev", "test")}

    for level in levels:
        log(f"seq/{config['scheme']}: training level '{level['name']}' ({model_config['head']} head)")
        marked = level.get("per_trigger", False)
        if marked and train_inputs == "predicted":
            train_items = _predicted_training_items(data["train"], stage1["train"], marked_helper, level)
            es_items, es_owners = _marked_queries(es_dev, stage1["es_dev"], marked_helper)
        elif marked:
            train_items = _marked_training_items(data["train"], marked_helper, level)
            es_items, es_owners = _marked_queries(es_dev, _gold_triggers(es_dev), marked_helper)
        else:
            train_items = _build_items(data["train"], encoder_helper, level, with_labels=True)
            es_items, es_owners = _build_items(es_dev, encoder_helper, level, with_labels=False), None

        set_seed(seed)  # the new head is initialised with this seed
        encoder = load_encoder(model_config["name"], **model_kwargs(model_config))
        if marked:
            encoder.resize_token_embeddings(len(tokenizer))   # rows for [TRG] and [/TRG]
        model = TokenTagger(
            encoder,
            level["tags"],
            head=model_config.get("head", "linear"),
            dropout=model_config.get("dropout", 0.1),
            mlp_hidden_size=model_config.get("mlp_hidden_size", 256),
            mlp_dropout=model_config.get("mlp_dropout", 0.1),
            mlp_activation=model_config.get("mlp_activation", "gelu"),
            crf_constrained=model_config.get("crf_constrained", True),
        )

        level_min_triggers = min_triggers if level["name"] == "trigger" else 0

        def evaluate_dev(current_model, level=level, es_items=es_items, es_owners=es_owners):
            # per-trigger levels: early stopping on the dev triggers (gold or predicted by
            # the trigger level, as in training)
            emissions = _emissions(current_model, es_items, collate, settings["eval_batch_size"], settings["precision"])
            strategy = native_decoding(current_model)
            if es_owners is None:
                preds = _decode_predictions(current_model, es_dev, es_items, emissions, level, strategy, level_min_triggers)
            else:
                preds = _marked_predictions(current_model, es_dev, es_items, es_owners, emissions, level, strategy)
            records = [{"id": s["id"], "pred": p} for s, p in zip(es_dev, preds)]
            return score_families(es_dev, records, level["families"])

        model, report = train_model(
            model, train_items, collate, evaluate_dev, settings, seed, name=f"seq_{level['name']}"
        )
        report["parameters"] = count_parameters(model)
        recorder.add_stage(report)

        if per_trigger and level["name"] == "trigger" and train_inputs == "predicted":
            # triggers of the training and early-stopping sentences for the argument levels
            for name, samples in (("train", data["train"]), ("es_dev", es_dev)):
                items = _build_items(samples, encoder_helper, level, with_labels=False)
                emissions = _emissions(model, items, collate, settings["eval_batch_size"], settings["precision"])
                preds = _decode_predictions(model, samples, items, emissions, level, native_decoding(model), min_triggers)
                stage1[name] = [_predicted_triggers(sample_preds) for sample_preds in preds]
            found = sum(len(t) for t in stage1["train"])
            gold = sum(len(trigger_groups(sample)) for sample in data["train"])
            log(f"seq/per_trigger: {found} triggers predicted on train ({gold} gold), argument levels trained on them")

        for split in ("dev", "test"):
            if marked:
                # one pass per trigger predicted by the trigger level (with the same decoding)
                for strategy in decodings:
                    with GpuMemory() as memory, Timer() as timer:
                        triggers = [_predicted_triggers(preds) for preds in merged[split][strategy]]
                        items, owners = _marked_queries(data[split], triggers, marked_helper)
                        emissions = _emissions(model, items, collate, settings["eval_batch_size"], settings["precision"])
                        preds = _marked_predictions(model, data[split], items, owners, emissions, level, strategy)
                    for index, sample_preds in enumerate(preds):
                        merged[split][strategy][index].extend(sample_preds)
                    _add_inference(inference[split][strategy], timer.seconds, memory.peak_gb)
                continue
            # timed from raw text (tokenization included), as in the QA pipelines
            with GpuMemory() as memory, Timer() as forward_timer:
                items = _build_items(data[split], encoder_helper, level, with_labels=False)
                emissions = _emissions(model, items, collate, settings["eval_batch_size"], settings["precision"])
            for strategy in decodings:
                with Timer() as decode_timer:
                    preds = _decode_predictions(model, data[split], items, emissions, level, strategy, level_min_triggers)
                for index, sample_preds in enumerate(preds):
                    merged[split][strategy][index].extend(sample_preds)
                _add_inference(inference[split][strategy], forward_timer.seconds + decode_timer.seconds, memory.peak_gb)
        del model, encoder
        cleanup()

    results = {}
    for strategy in decodings:
        variant = f"decoding={strategy}"
        for split in ("dev", "test"):
            # two_level / authors_flat tag spans only: events are rebuilt by the nearest-trigger
            # rule; per_trigger already attached every argument to its trigger's event
            records = [
                {"id": sample["id"], "pred": preds if per_trigger else assign_by_nearest_trigger(preds)}
                for sample, preds in zip(data[split], merged[split][strategy])
            ]
            metrics = recorder.add_variant(variant, split, data[split], records, inference[split][strategy])
            results[(variant, split)] = metrics["summary"]
    return recorder.finish()

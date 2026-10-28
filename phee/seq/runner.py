"""Sequence labeling experiment: train, predict with every decoding, evaluate.

Config (configs/seq/*.yaml):
  method: seq
  scheme: two_level | authors_flat
  model:
    name: thomas-sounack/BioClinical-ModernBERT-base
    head: linear | mlp | crf
    dropout, mlp_hidden_size, mlp_dropout, mlp_activation, crf_constrained, max_length
  training: see phee.training.DEFAULT_TRAINING (+ crf_learning_rate)
  decodings: [argmax, viterbi]
"""

from collections import defaultdict

import torch

from phee.evaluation import score_families
from phee.experiment import RunRecorder, early_stopping_dev, load_splits
from phee.hf import count_parameters, load_encoder, load_tokenizer, model_kwargs
from phee.seq.decoding import decode, native_decoding
from phee.seq.encoding import WordEncoder, make_collate
from phee.seq.model import TokenTagger
from phee.seq.tagging import (
    AUTHORS_TAGS,
    OWN_MAIN_TAGS,
    OWN_SUB_TAGS,
    authors_tags_to_entities,
    authors_training_sequences,
    bio_to_entities,
    entity_to_prediction,
    own_main_tags,
    own_sub_tags,
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


def _decode_predictions(model, samples, items, emissions, level, strategy):
    """Predictions (unified format) of one level for every sample."""
    predictions = []
    index = 0
    for scores, word_mask in emissions:
        for row in decode(model, scores, word_mask, strategy):
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
            predictions.append(preds)
            index += 1
    return predictions


def run(config, seed, run_dir):
    data = load_splits(config)
    model_config = config["model"]
    settings = training_settings(config.get("training"))
    decodings = config.get("decodings", ["argmax"])
    levels = _levels(config["scheme"])
    recorder = RunRecorder(run_dir, config, seed)
    es_dev = early_stopping_dev(config, data)

    tokenizer = load_tokenizer(model_config["name"])
    encoder_helper = WordEncoder(tokenizer, max_length=model_config.get("max_length", 512))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate = make_collate(pad_id)

    # predictions[split][decoding] -> list (per sample) of prediction lists, merged over levels
    merged = {split: {d: [[] for _ in data[split]] for d in decodings} for split in ("dev", "test")}
    inference = {split: defaultdict(lambda: {"seconds": 0.0, "peak_gpu_memory_gb": None}) for split in ("dev", "test")}

    for level in levels:
        log(f"seq/{config['scheme']}: training level '{level['name']}' ({model_config['head']} head)")
        train_items = _build_items(data["train"], encoder_helper, level, with_labels=True)
        es_items = _build_items(es_dev, encoder_helper, level, with_labels=False)

        set_seed(seed)  # the new head is initialised with this seed
        encoder = load_encoder(model_config["name"], **model_kwargs(model_config))
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

        def evaluate_dev(current_model, level=level):
            emissions = _emissions(current_model, es_items, collate, settings["eval_batch_size"], settings["precision"])
            preds = _decode_predictions(
                current_model, es_dev, es_items, emissions, level, native_decoding(current_model)
            )
            records = [{"id": s["id"], "pred": p} for s, p in zip(es_dev, preds)]
            return score_families(es_dev, records, level["families"])

        model, report = train_model(
            model, train_items, collate, evaluate_dev, settings, seed, name=f"seq_{level['name']}"
        )
        report["parameters"] = count_parameters(model)
        recorder.add_stage(report)

        for split in ("dev", "test"):
            # timed from raw text (tokenization included), as in the QA pipelines
            with GpuMemory() as memory, Timer() as forward_timer:
                items = _build_items(data[split], encoder_helper, level, with_labels=False)
                emissions = _emissions(model, items, collate, settings["eval_batch_size"], settings["precision"])
            for strategy in decodings:
                with Timer() as decode_timer:
                    preds = _decode_predictions(model, data[split], items, emissions, level, strategy)
                for index, sample_preds in enumerate(preds):
                    merged[split][strategy][index].extend(sample_preds)
                record = inference[split][strategy]
                record["seconds"] += forward_timer.seconds + decode_timer.seconds
                record["peak_gpu_memory_gb"] = max(
                    [value for value in (record["peak_gpu_memory_gb"], memory.peak_gb) if value is not None],
                    default=None,
                )
        del model, encoder
        cleanup()

    results = {}
    for strategy in decodings:
        variant = f"decoding={strategy}"
        for split in ("dev", "test"):
            records = [
                {"id": sample["id"], "pred": preds}
                for sample, preds in zip(data[split], merged[split][strategy])
            ]
            metrics = recorder.add_variant(variant, split, data[split], records, inference[split][strategy])
            results[(variant, split)] = metrics["summary"]
    return recorder.finish()

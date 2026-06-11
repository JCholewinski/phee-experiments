# scripts/train_event_type_classifier.py

import argparse
import json
import time
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


EVENT_LABELS = ["ADE", "PTE"]

RAW_LABEL_TO_EVENT_TYPE = {
    "Adverse_event": "ADE",
    "Potential_therapeutic_event": "PTE",
}


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def sample_to_record(sample, record_index):
    tokens = sample["sentence"]
    text = " ".join(str(token) for token in tokens)

    event_types = set()

    for event in sample["event"]:
        for _, _, raw_label in event:
            if raw_label in RAW_LABEL_TO_EVENT_TYPE:
                event_types.add(RAW_LABEL_TO_EVENT_TYPE[raw_label])

    labels = [
        1.0 if event_label in event_types else 0.0
        for event_label in EVENT_LABELS
    ]

    return {
        "record_index": record_index,
        "id": sample.get("id", record_index),
        "text": text,
        "labels": labels,
    }


def build_dataset(path):
    samples = load_jsonl(path)
    records = [
        sample_to_record(sample, record_index=i)
        for i, sample in enumerate(samples)
    ]
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/event_type_classifier.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    max_length = config["training"].get("max_length", 256)

    train_dataset = build_dataset(config["data"]["train_path"])
    val_dataset = build_dataset(config["data"]["val_path"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = batch["labels"]
        return tokenized

    train_tokenized = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    val_tokenized = val_dataset.map(
        tokenize,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EVENT_LABELS),
        id2label={idx: label for idx, label in enumerate(EVENT_LABELS)},
        label2id={label: idx for idx, label in enumerate(EVENT_LABELS)},
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        num_train_epochs=config["training"]["num_train_epochs"],
        weight_decay=config["training"]["weight_decay"],
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
    )

    start = time.perf_counter()
    trainer.train()
    end = time.perf_counter()

    final_output_dir = config["training"]["output_dir_final"]

    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    Path("outputs/timing").mkdir(parents=True, exist_ok=True)

    timing = {
        "method": "event_type_classifier",
        "train_runtime_seconds": end - start,
        "model_name": model_name,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "output_dir_final": final_output_dir,
    }

    with open(
        "outputs/timing/event_type_classifier_training_timing.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    print(json.dumps(timing, indent=2, ensure_ascii=False))
    print(f"Saved final model to: {final_output_dir}")


if __name__ == "__main__":
    main()
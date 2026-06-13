# scripts/train_extractive_qa_trigger.py

import argparse
import json
import time
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def prepare_train_features(examples, tokenizer, max_length, doc_stride):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]

        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        answer_start = answers["answer_start"][0]
        answer_text = answers["text"][0]
        answer_end = answer_start + len(answer_text)

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (
            offsets[token_start_index][0] <= answer_start
            and offsets[token_end_index][1] >= answer_end
        ):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= answer_start
            ):
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while offsets[token_end_index][1] >= answer_end:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions

    return tokenized_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/extractive_qa_subarguments.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Using config: {args.config}")
    print(f"Model: {config['model']['name']}")
    print(f"Output dir: {config['training']['output_dir']}")
    print(f"Final output dir: {config['training']['output_dir_final']}")

    train_records = load_jsonl(config["data"]["train_path"])
    val_records = load_jsonl(config["data"]["val_path"])

    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    model = AutoModelForQuestionAnswering.from_pretrained(config["model"]["name"])

    max_length = config["training"].get("max_length", 384)
    doc_stride = config["training"].get("doc_stride", 128)

    train_tokenized = train_dataset.map(
        lambda examples: prepare_train_features(
            examples=examples,
            tokenizer=tokenizer,
            max_length=max_length,
            doc_stride=doc_stride,
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    val_tokenized = val_dataset.map(
        lambda examples: prepare_train_features(
            examples=examples,
            tokenizer=tokenizer,
            max_length=max_length,
            doc_stride=doc_stride,
        ),
        batched=True,
        remove_columns=val_dataset.column_names,
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
        "method": "extractive_qa_subarguments",
        "train_runtime_seconds": end - start,
        "model_name": config["model"]["name"],
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "output_dir_final": final_output_dir,
    }

    timing_output_path = Path("outputs/timing/extractive_qa_subarguments_training_timing.json")

    with open(timing_output_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    print(json.dumps(timing, indent=2, ensure_ascii=False))
    print(f"Saved final model to: {final_output_dir}")


if __name__ == "__main__":
    main()
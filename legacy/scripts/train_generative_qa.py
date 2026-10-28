import argparse
import json
import time
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.preprocessing.to_generative_qa import load_jsonl


def build_dataset(path):
    return Dataset.from_list(load_jsonl(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method_name", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    max_source_length = config["training"].get("max_source_length", 512)
    max_target_length = config["training"].get("max_target_length", 256)

    train_dataset = build_dataset(config["data"]["train_path"])
    val_dataset = build_dataset(config["data"]["val_path"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_source_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    train_tokenized = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    val_tokenized = val_dataset.map(
        preprocess,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["training"]["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        num_train_epochs=config["training"]["num_train_epochs"],
        weight_decay=config["training"]["weight_decay"],
        predict_with_generate=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
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
        "method": args.method_name,
        "model_name": model_name,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "train_runtime_seconds": end - start,
        "output_dir_final": final_output_dir,
    }

    timing_path = Path(f"outputs/timing/{args.method_name}_training_timing.json")

    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    print(json.dumps(timing, indent=2, ensure_ascii=False))
    print(f"Saved final model to: {final_output_dir}")


if __name__ == "__main__":
    main()

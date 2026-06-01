import argparse
import json
import time
from pathlib import Path

import yaml
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer

from src.datasets.seq_dataset import build_dataset
from src.models import BertFrozenLinearCRFForTokenClassification
from src.preprocessing.to_bio import convert_sample_to_bio
from src.preprocessing.tokenize_and_align_crf import tokenize_and_align_crf
from src.utils.labeling import LABEL_LIST, LABEL2ID, ID2LABEL


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/seq_linear_crf_frozen.yaml",
    )
    parser.add_argument(
        "--linear_model_path",
        type=str,
        default=None,
        help="Path to trained linear model, e.g. outputs/seq/linear/final",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    linear_model_path = args.linear_model_path or config["model"]["name"]
    train_path = config["data"]["train_path"]

    raw_data = load_jsonl(train_path)
    dataset = build_dataset(raw_data, convert_sample_to_bio)

    tokenizer = AutoTokenizer.from_pretrained(linear_model_path)

    dataset = dataset.map(
        lambda x: tokenize_and_align_crf(x, tokenizer, LABEL2ID),
        batched=False,
    )

    model_config = AutoConfig.from_pretrained(
        linear_model_path,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model = BertFrozenLinearCRFForTokenClassification.from_pretrained(
        linear_model_path,
        config=model_config,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Loaded trained linear model from: {linear_model_path}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        num_train_epochs=config["training"]["num_train_epochs"],
        logging_steps=config["training"]["logging_steps"],
        learning_rate=config["training"]["learning_rate"],
        save_strategy=config["training"]["save_strategy"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    start_time = time.perf_counter()
    trainer.train()
    end_time = time.perf_counter()

    output_dir_final = config["training"]["output_dir_final"]
    trainer.save_model(output_dir_final)
    tokenizer.save_pretrained(output_dir_final)

    metrics_path = Path("outputs/metrics/linear_crf_frozen_training_time.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "linear_crf_frozen",
                "base_model": linear_model_path,
                "training_time_seconds": end_time - start_time,
                "training_time_minutes": (end_time - start_time) / 60,
                "num_train_samples": len(dataset),
                "num_epochs": training_args.num_train_epochs,
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved Frozen Linear + CRF model to: {output_dir_final}")


if __name__ == "__main__":
    main()
import json
import yaml
import os
import argparse
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, TrainingArguments, Trainer
from src.preprocessing.to_bio import convert_sample_to_bio
from src.models import BertMLPForTokenClassification, BertCRFForTokenClassification
from src.preprocessing.tokenize_and_align import tokenize_and_align
from src.preprocessing.tokenize_and_align_crf import tokenize_and_align_crf 
from src.utils.labeling import LABEL_LIST, LABEL2ID, ID2LABEL
from src.datasets.seq_dataset import load_dataset, build_dataset
from huggingface_hub import upload_folder, create_repo



parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="configs/seq.yaml",
    help="Path to the YAML configuration file.",
)
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
HEAD_TYPE = config["model"].get("head_type", "linear")
TRAIN_PATH = config["data"]["train_path"]

# 1. load data
raw_data = []
with open(TRAIN_PATH) as f:
    for line in f:
        raw_data.append(json.loads(line))

# ograniczenie w celu debugowania
# raw_data = raw_data[:200]

# 2. BIO
dataset = build_dataset(raw_data, convert_sample_to_bio)

# 3. tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


if HEAD_TYPE == "crf":
    dataset = dataset.map(
        lambda x: tokenize_and_align_crf(x, tokenizer, LABEL2ID),
        batched=False,
    )
else:
    dataset = dataset.map(
        lambda x: tokenize_and_align(x, tokenizer, LABEL2ID),
        batched=False,
    )

# 4. model
if HEAD_TYPE == "linear":
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

elif HEAD_TYPE == "mlp":
    model_config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model_config.mlp_hidden_size = config["model"].get(
        "mlp_hidden_size",
        model_config.hidden_size,
    )
    model_config.mlp_dropout = config["model"].get(
        "mlp_dropout",
        model_config.hidden_dropout_prob,
    )
    model_config.mlp_activation = config["model"].get(
        "mlp_activation",
        "gelu",
    )

    model = BertMLPForTokenClassification.from_pretrained(
        MODEL_NAME,
        config=model_config,
    )
elif HEAD_TYPE == "crf":
    model_config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model_config.crf_ce_loss_weight = config["model"].get("crf_ce_loss_weight", 1.0)
    model_config.crf_loss_weight = config["model"].get("crf_loss_weight", 0.1)

    model = BertCRFForTokenClassification.from_pretrained(
        MODEL_NAME,
        config=model_config,
    )

else:
    raise ValueError(f"Unsupported head_type: {HEAD_TYPE}")

# 5. training
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
    processing_class=tokenizer
)

start_time = time.perf_counter()

trainer.train()

end_time = time.perf_counter()
training_time_seconds = end_time - start_time

timing_output_path = Path("outputs/metrics/seq_{HEAD_TYPE}_training_time.json")
timing_output_path.parent.mkdir(parents=True, exist_ok=True)

with open(timing_output_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "method": f"sequence_labeling_{HEAD_TYPE}",
            "training_time_seconds": training_time_seconds,
            "training_time_minutes": training_time_seconds / 60,
            "num_train_samples": len(dataset),
            "num_epochs": training_args.num_train_epochs,
            "model_name": config["model"]["name"],
            "head_type": HEAD_TYPE,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"Training time: {training_time_seconds:.2f} seconds")

trainer.save_model(config["training"]["output_dir_final"])
tokenizer.save_pretrained(config["training"]["output_dir_final"])

upload_to_hf = os.getenv("UPLOAD_TO_HF", "false").lower() == "true"

if upload_to_hf:
    create_repo("jcholewinski/sequential_first_model", exist_ok=True)

    upload_folder(
        folder_path=config["training"]["output_dir_final"],
        repo_id="jcholewinski/sequential_first_model",
        repo_type="model",
    )

    print("Model uploaded to Hugging Face.")
else:
    print("Skipping Hugging Face upload because UPLOAD_TO_HF=false.")
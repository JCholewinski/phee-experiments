import json
import yaml
import os
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from src.preprocessing.to_bio import convert_sample_to_bio
from src.preprocessing.tokenize_and_align import tokenize_and_align
from src.utils.labeling import LABEL_LIST, LABEL2ID, ID2LABEL
from src.datasets.seq_dataset import load_dataset, build_dataset
from huggingface_hub import upload_folder, create_repo


with open("configs/seq.yaml") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
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

dataset = dataset.map(
    lambda x: tokenize_and_align(x, tokenizer, LABEL2ID),
    batched=False
)

# 4. model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

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

timing_output_path = Path("outputs/metrics/seq_training_time.json")
timing_output_path.parent.mkdir(parents=True, exist_ok=True)

with open(timing_output_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "method": "sequence_labeling",
            "training_time_seconds": training_time_seconds,
            "training_time_minutes": training_time_seconds / 60,
            "num_train_samples": len(dataset),
            "num_epochs": training_args.num_train_epochs,
            "model_name": config["model"]["name"],
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
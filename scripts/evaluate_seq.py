import json
import os
import yaml
import torch
import numpy as np
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score

from src.preprocessing.to_bio import convert_sample_to_bio
from src.preprocessing.tokenize_and_align import tokenize_and_align
from src.utils.labeling import LABEL2ID, ID2LABEL


# ======================
# 1. config
# ======================
with open("configs/seq.yaml") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["name"]
TEST_PATH = config["data"]["test_path"]
MODEL_PATH = config["training"]["output_dir_final"]


# ======================
# 2. load JSONL
# ======================
data = []
with open(TEST_PATH) as f:
    for line in f:
        data.append(json.loads(line))


# ======================
# 3. convert → BIO
# ======================
processed = [convert_sample_to_bio(x) for x in data]
dataset = Dataset.from_list(processed)


# ======================
# 4. tokenizer + alignment
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = dataset.map(
    lambda x: tokenize_and_align(x, tokenizer, LABEL2ID),
    batched=False
)


# ======================
# 5. model
# ======================
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()


# ======================
# 6. inference
# ======================
all_preds = []
all_labels = []

for sample in dataset:

    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0)
    labels = sample["labels"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)[0].numpy()

    true_labels = []
    pred_labels = []

    for p, l in zip(preds, labels):
        if l == -100:
            continue
        true_labels.append(ID2LABEL[l])
        pred_labels.append(ID2LABEL[p])

    all_labels.append(true_labels)
    all_preds.append(pred_labels)


# ======================
# 7. metryki
# ======================
f1 = f1_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds)

print("\n=== F1 ===")
print(f1)

print("\n=== CLASSIFICATION REPORT ===")
print(report)


# ======================
# 8. zapis wyników
# ======================
os.makedirs(MODEL_PATH, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# JSON (łatwy do dalszego użycia)
metrics_path = os.path.join(MODEL_PATH, f"metrics_{timestamp}.json")
with open(metrics_path, "w") as f:
    json.dump({"f1": f1}, f, indent=2)

# tekstowy raport
report_path = os.path.join(MODEL_PATH, f"report_{timestamp}.txt")
with open(report_path, "w") as f:
    f.write(report)

print(f"\nSaved metrics to: {metrics_path}")
print(f"Saved report to: {report_path}")
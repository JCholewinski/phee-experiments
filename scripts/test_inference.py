import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.utils.labeling import ID2LABEL


MODEL_PATH = "/Users/jakubcholewinski/Coding/Magisterka/phee-experiments/outputs/seq/checkpoint-50"  

# 1. load
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

model.eval()

# 2. test sentence
tokens = ["Patient", "took", "medicine", "which", "name", "is", "aspirine", "and", "developed", "headache"]

# 3. tokenize
inputs = tokenizer(
    tokens,
    is_split_into_words=True,
    return_tensors="pt"
)

# 4. model
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)[0]

# 5. map to labels
word_ids = inputs.word_ids()

labels = []
for word_id, pred_id in zip(word_ids, predictions):
    if word_id is None:
        continue
    labels.append(ID2LABEL[pred_id.item()])

# 6. print
print("TOKENS:", [[word, label] for word, label in zip(tokens, labels)])

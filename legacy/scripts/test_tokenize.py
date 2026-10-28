import json
from transformers import AutoTokenizer

from src.preprocessing.to_bio import convert_sample_to_bio
from src.preprocessing.tokenize_and_align import tokenize_and_align
from src.utils.labeling import LABEL2ID

data = []
with open("data/raw/train_main.json") as f:
    for line in f:
        data.append(json.loads(line))

sample = convert_sample_to_bio(data[0])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

result = tokenize_and_align(sample, tokenizer, LABEL2ID)

print(len(result["input_ids"]))
print(len(result["labels"]))
print(result["labels"][:20])
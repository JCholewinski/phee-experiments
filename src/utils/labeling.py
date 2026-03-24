LABEL_LIST = [
    "O",
    "B-TRIGGER", "I-TRIGGER",
    "B-EFFECT", "I-EFFECT",
    "B-TREATMENT", "I-TREATMENT"
]

LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
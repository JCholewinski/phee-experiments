LABEL_LIST = [
    "O",

    "B-ADE_TRIGGER",
    "I-ADE_TRIGGER",

    "B-PTE_TRIGGER",
    "I-PTE_TRIGGER",

    "B-SUBJECT",
    "I-SUBJECT",

    "B-TREATMENT",
    "I-TREATMENT",

    "B-EFFECT",
    "I-EFFECT",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
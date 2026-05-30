from collections import defaultdict


SUBARGUMENT_PRIORITY = {
    "Combination.Drug": 1,

    "Treatment.Drug": 2,
    "Treatment.Dosage": 3,
    "Treatment.Route": 4,
    "Treatment.Freq": 5,
    "Treatment.Duration": 6,
    "Treatment.Time_elapsed": 7,
    "Treatment.Disorder": 8,

    "Subject.Age": 9,
    "Subject.Gender": 10,
    "Subject.Race": 11,
    "Subject.Population": 12,
    "Subject.Disorder": 13,
}


def map_subargument_label(label):
    mapping = {
        "Combination.Drug": "COMBINATION_DRUG",

        "Treatment.Drug": "TREATMENT_DRUG",
        "Treatment.Dosage": "TREATMENT_DOSAGE",
        "Treatment.Route": "TREATMENT_ROUTE",
        "Treatment.Freq": "TREATMENT_FREQ",
        "Treatment.Duration": "TREATMENT_DURATION",
        "Treatment.Time_elapsed": "TREATMENT_TIME_ELAPSED",
        "Treatment.Disorder": "TREATMENT_DISORDER",

        "Subject.Age": "SUBJECT_AGE",
        "Subject.Gender": "SUBJECT_GENDER",
        "Subject.Race": "SUBJECT_RACE",
        "Subject.Population": "SUBJECT_POPULATION",
        "Subject.Disorder": "SUBJECT_DISORDER",
    }

    return mapping.get(label, "O")


def choose_label(raw_labels):
    if not raw_labels:
        return "O"

    return min(
        raw_labels,
        key=lambda label: SUBARGUMENT_PRIORITY.get(label, 999)
    )


def clean_bio_tokens(tokens, labels):
    cleaned_tokens = []
    cleaned_labels = []

    for token, label in zip(tokens, labels):
        if token is None:
            continue

        token = str(token)

        if token.strip() == "":
            continue

        cleaned_tokens.append(token)
        cleaned_labels.append(label)

    fixed_labels = []
    previous_entity = None

    for label in cleaned_labels:
        if label == "O":
            fixed_labels.append("O")
            previous_entity = None
            continue

        prefix, entity = label.split("-", 1)

        if prefix == "I" and previous_entity != entity:
            fixed_labels.append(f"B-{entity}")
        else:
            fixed_labels.append(label)

        previous_entity = entity

    return cleaned_tokens, fixed_labels


def convert_sample_to_bio_subarguments(sample):
    tokens = sample["sentence"]

    token_to_raw_labels = defaultdict(list)

    for event in sample["event"]:
        for start, end, raw_label in event:
            if raw_label not in SUBARGUMENT_PRIORITY:
                continue

            for token_idx in range(start, end + 1):
                token_to_raw_labels[token_idx].append(raw_label)

    chosen_labels = ["O"] * len(tokens)

    for token_idx, raw_labels in token_to_raw_labels.items():
        chosen_raw_label = choose_label(raw_labels)
        chosen_label = map_subargument_label(chosen_raw_label)
        chosen_labels[token_idx] = chosen_label

    bio_labels = ["O"] * len(tokens)

    previous_label = "O"

    for idx, label in enumerate(chosen_labels):
        if label == "O":
            bio_labels[idx] = "O"
            previous_label = "O"
            continue

        if previous_label == label:
            bio_labels[idx] = f"I-{label}"
        else:
            bio_labels[idx] = f"B-{label}"

        previous_label = label

    tokens, bio_labels = clean_bio_tokens(tokens, bio_labels)

    return {
        "id": sample.get("id"),
        "tokens": tokens,
        "labels": bio_labels,
    }

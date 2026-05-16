def map_label(label):

    if label.startswith("Adverse_event"):
        return "TRIGGER"
    elif label.startswith("Effect"):
        return "EFFECT"
    elif label.startswith("Treatment"):
        return "TREATMENT"
    else:
        return "O"


def convert_sample_to_bio(sample):

    tokens = sample["sentence"]
    labels = ["O"] * len(tokens)

    for event in sample["event"]:
        for start, end, label in event:

            mapped = map_label(label)

            if mapped == "O":
                continue

            for i in range(start, end + 1):
                if i == start:
                    labels[i] = f"B-{mapped}"
                else:
                    labels[i] = f"I-{mapped}"

    tokens, labels = clean_bio_tokens(tokens, labels)

    return {
        "tokens": tokens,
        "labels": labels,
    }

def clean_bio_tokens(tokens, labels):
    """
    Remove empty or whitespace-only tokens from BIO representation.
    Also fixes invalid I-* labels after removal if necessary.
    """

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
            fixed_labels.append(label)
            previous_entity = None
            continue

        if "-" not in label:
            fixed_labels.append(label)
            previous_entity = None
            continue

        prefix, entity = label.split("-", 1)

        if prefix == "I" and previous_entity != entity:
            fixed_labels.append(f"B-{entity}")
        else:
            fixed_labels.append(label)

        previous_entity = entity

    return cleaned_tokens, fixed_labels
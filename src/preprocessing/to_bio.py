def map_label(label):
    if label == "Adverse_event":
        return "ADE_TRIGGER"

    if label == "Potential_therapeutic_event":
        return "PTE_TRIGGER"

    if label == "Effect":
        return "EFFECT"

    if label == "Treatment":
        return "TREATMENT"

    return "O"


def merge_spans(spans):
    spans = sorted(spans, key=lambda x: (x["start"], x["end"]))

    merged = []

    for span in spans:
        if not merged:
            merged.append(span)
            continue

        last = merged[-1]

        same_label = last["label"] == span["label"]
        overlaps_or_touches = span["start"] <= last["end"] + 1

        if same_label and overlaps_or_touches:
            last["end"] = max(last["end"], span["end"])
        else:
            merged.append(span)

    return merged


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


def convert_sample_to_bio(sample):
    tokens = sample["sentence"]

    coarse_spans = []

    for event in sample["event"]:
        for start, end, raw_label in event:
            label = map_label(raw_label)

            if label == "O":
                continue

            coarse_spans.append({
                "start": start,
                "end": end,
                "label": label,
            })

    coarse_spans = merge_spans(coarse_spans)

    labels = ["O"] * len(tokens)

    for span in coarse_spans:
        start = span["start"]
        end = span["end"]
        label = span["label"]

        labels[start] = f"B-{label}"

        for i in range(start + 1, end + 1):
            labels[i] = f"I-{label}"

    tokens, labels = clean_bio_tokens(tokens, labels)

    return {
        "tokens": tokens,
        "labels": labels,
    }
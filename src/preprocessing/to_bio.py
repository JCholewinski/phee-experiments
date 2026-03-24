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

    return {
        "tokens": tokens,
        "labels": labels
    }
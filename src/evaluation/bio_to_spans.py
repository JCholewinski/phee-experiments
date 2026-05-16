# src/evaluation/bio_to_spans.py

from typing import List, Dict, Any


def bio_to_spans(tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
    """
    Convert BIO labels into span-level representation.

    Example:
        tokens = ["patient", "received", "aspirin"]
        labels = ["O", "O", "B-TREATMENT"]

    Returns:
        [
            {
                "label": "TREATMENT",
                "text": "aspirin",
                "start": 2,
                "end": 2
            }
        ]
    """

    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must have the same length")

    spans = []
    current = None

    for i, label in enumerate(labels):
        if label == "O" or label is None:
            if current is not None:
                current["end"] = i - 1
                current["text"] = " ".join(tokens[current["start"]:current["end"] + 1])
                spans.append(current)
                current = None
            continue

        if "-" not in label:
            raise ValueError(f"Invalid BIO label: {label}")

        prefix, span_label = label.split("-", 1)

        if prefix == "B":
            if current is not None:
                current["end"] = i - 1
                current["text"] = " ".join(tokens[current["start"]:current["end"] + 1])
                spans.append(current)

            current = {
                "label": span_label,
                "start": i,
                "end": i,
                "text": tokens[i],
            }

        elif prefix == "I":
            if current is None or current["label"] != span_label:
                current = {
                    "label": span_label,
                    "start": i,
                    "end": i,
                    "text": tokens[i],
                }

        else:
            raise ValueError(f"Invalid BIO prefix: {prefix}")

    if current is not None:
        current["end"] = len(tokens) - 1
        current["text"] = " ".join(tokens[current["start"]:current["end"] + 1])
        spans.append(current)

    return spans
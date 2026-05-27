# src/evaluation/bio_to_spans.py

def normalize_span_label(raw_label):
    if raw_label == "ADE_TRIGGER":
        return "TRIGGER", "ADE"

    if raw_label == "PTE_TRIGGER":
        return "TRIGGER", "PTE"

    return raw_label, None


def make_span(tokens, raw_label, start_idx):
    normalized_label, event_type = normalize_span_label(raw_label)

    span = {
        "label": normalized_label,
        "_raw_label": raw_label,
        "start": start_idx,
        "end": start_idx,
        "text": tokens[start_idx],
    }

    if event_type is not None:
        span["event_type"] = event_type

    return span


def close_span(span):
    if span is None:
        return None

    span = dict(span)
    span.pop("_raw_label", None)
    return span


def bio_to_spans(tokens, labels):
    spans = []
    current = None

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == "O":
            closed = close_span(current)
            if closed is not None:
                spans.append(closed)
            current = None
            continue

        if "-" not in label:
            closed = close_span(current)
            if closed is not None:
                spans.append(closed)

            current = make_span(tokens, label, i)
            continue

        prefix, raw_label = label.split("-", 1)

        if prefix == "B":
            closed = close_span(current)
            if closed is not None:
                spans.append(closed)

            current = make_span(tokens, raw_label, i)

        elif prefix == "I":
            if current is None or current["_raw_label"] != raw_label:
                closed = close_span(current)
                if closed is not None:
                    spans.append(closed)

                current = make_span(tokens, raw_label, i)
            else:
                current["end"] = i
                current["text"] = " ".join(tokens[current["start"]:i + 1])

        else:
            closed = close_span(current)
            if closed is not None:
                spans.append(closed)
            current = None

    closed = close_span(current)
    if closed is not None:
        spans.append(closed)

    return spans
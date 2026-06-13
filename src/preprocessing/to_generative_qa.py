import difflib
import json
import re
from pathlib import Path


TRIGGER_LABEL_TO_EVENT_TYPE = {
    "Adverse_event": "ADE",
    "Potential_therapeutic_event": "PTE",
}

RAW_MAIN_ARGUMENT_TO_LABEL = {
    "Subject": "SUBJECT",
    "Treatment": "TREATMENT",
    "Effect": "EFFECT",
}

RAW_SUBARGUMENT_TO_LABEL = {
    "Subject.Age": "SUBJECT_AGE",
    "Subject.Gender": "SUBJECT_GENDER",
    "Subject.Race": "SUBJECT_RACE",
    "Subject.Population": "SUBJECT_POPULATION",
    "Subject.Disorder": "SUBJECT_DISORDER",

    "Treatment.Drug": "TREATMENT_DRUG",
    "Treatment.Dosage": "TREATMENT_DOSAGE",
    "Treatment.Route": "TREATMENT_ROUTE",
    "Treatment.Freq": "TREATMENT_FREQ",
    "Treatment.Duration": "TREATMENT_DURATION",
    "Treatment.Time_elapsed": "TREATMENT_TIME_ELAPSED",
    "Treatment.Disorder": "TREATMENT_DISORDER",

    "Combination.Drug": "COMBINATION_DRUG",
}

SUBARGUMENT_LABELS_BY_MAIN_ARGUMENT = {
    "SUBJECT": [
        "SUBJECT_AGE",
        "SUBJECT_GENDER",
        "SUBJECT_RACE",
        "SUBJECT_POPULATION",
        "SUBJECT_DISORDER",
    ],
    "TREATMENT": [
        "TREATMENT_DRUG",
        "TREATMENT_DOSAGE",
        "TREATMENT_ROUTE",
        "TREATMENT_FREQ",
        "TREATMENT_DURATION",
        "TREATMENT_TIME_ELAPSED",
        "TREATMENT_DISORDER",
        "COMBINATION_DRUG",
    ],
}

SUBARGUMENT_QUESTIONS = {
    "SUBJECT_AGE": 'What is the age of the subject "{main_argument_text}"?',
    "SUBJECT_GENDER": 'What is the gender of the subject "{main_argument_text}"?',
    "SUBJECT_RACE": 'What is the race of the subject "{main_argument_text}"?',
    "SUBJECT_POPULATION": 'What population describes the subject "{main_argument_text}"?',
    "SUBJECT_DISORDER": 'What disorder describes the subject "{main_argument_text}"?',

    "TREATMENT_DRUG": 'What drug is mentioned in the treatment "{main_argument_text}"?',
    "TREATMENT_DOSAGE": 'What dosage is mentioned for the treatment "{main_argument_text}"?',
    "TREATMENT_ROUTE": 'What route is mentioned for the treatment "{main_argument_text}"?',
    "TREATMENT_FREQ": 'What frequency is mentioned for the treatment "{main_argument_text}"?',
    "TREATMENT_DURATION": 'What duration is mentioned for the treatment "{main_argument_text}"?',
    "TREATMENT_TIME_ELAPSED": 'What time elapsed is mentioned for the treatment "{main_argument_text}"?',
    "TREATMENT_DISORDER": 'What disorder is treated by "{main_argument_text}"?',
    "COMBINATION_DRUG": 'What drug is used in combination with the treatment "{main_argument_text}"?',
}

STAGE1_QUESTION = "What are the events?"

EVENT_MARKERS = {
    "ADE": "[ADE]",
    "PTE": "[PTE]",
}

MAIN_ARGUMENT_MARKERS = {
    "SUBJECT": "[Subject]",
    "TREATMENT": "[Treatment]",
    "EFFECT": "[Effect]",
}

MARKER_TO_EVENT_TYPE = {
    "[ade]": "ADE",
    "[adverseevent]": "ADE",
    "[adverse_event]": "ADE",
    "[pte]": "PTE",
    "[potentialtherapeuticevent]": "PTE",
    "[potential_therapeutic_event]": "PTE",
}

MARKER_TO_LABEL = {
    "[subject]": "SUBJECT",
    "[treatment]": "TREATMENT",
    "[effect]": "EFFECT",
}

MARKER_PATTERN = re.compile(
    r"(\[\s*ADE\s*\]|\[\s*PTE\s*\]|\[\s*Adverse[_\s]*Event\s*\]|"
    r"\[\s*Potential[_\s]*Therapeutic[_\s]*Event\s*\]|"
    r"\[\s*Subject\s*\]|\[\s*Treatment\s*\]|\[\s*Effect\s*\])",
    re.IGNORECASE,
)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tokens_to_text(tokens):
    return " ".join(str(t) for t in tokens)


def token_span_text(tokens, start, end):
    return " ".join(str(t) for t in tokens[start:end + 1])


def build_stage1_input(sample):
    return f"question: {STAGE1_QUESTION} context: {tokens_to_text(sample['sentence'])}"


def build_stage2_input(question, sample):
    return f"question: {question} context: {tokens_to_text(sample['sentence'])}"


def clean_piece(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"^[\s:;,\-\|]+", "", text)
    text = re.sub(r"[\s:;,\-\|]+$", "", text)
    return text.strip()


def normalize_for_match(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_no_answer(text):
    norm = normalize_for_match(text)
    return norm in {"", "none", "no answer", "noanswer", "null", "n a", "not applicable"}


def canonical_marker(marker):
    marker = re.sub(r"\s+", "", marker.lower())
    return marker


def find_token_span_for_text(answer_text, tokens, max_window=80, min_score=0.78):
    answer_norm = normalize_for_match(answer_text)

    if not answer_norm:
        return None, None

    n = len(tokens)
    best = None

    for i in range(n):
        upper = min(n, i + max_window)
        for j in range(i, upper):
            candidate = token_span_text(tokens, i, j)
            cand_norm = normalize_for_match(candidate)

            if not cand_norm:
                continue

            if cand_norm == answer_norm:
                return i, j

            if len(answer_norm) >= 4 and answer_norm in cand_norm:
                score = len(answer_norm) / max(len(cand_norm), 1)
            elif len(cand_norm) >= 4 and cand_norm in answer_norm:
                score = len(cand_norm) / max(len(answer_norm), 1)
            else:
                score = difflib.SequenceMatcher(None, answer_norm, cand_norm).ratio()

            if best is None or score > best[0]:
                best = (score, i, j)

    if best is None or best[0] < min_score:
        return None, None

    return best[1], best[2]


def get_event_type_from_event(event):
    for _, _, raw_label in event:
        if raw_label in TRIGGER_LABEL_TO_EVENT_TYPE:
            return TRIGGER_LABEL_TO_EVENT_TYPE[raw_label]
    return None


def serialize_stage1_target(sample):
    tokens = sample["sentence"]
    event_strings = []

    for event in sample["event"]:
        event_type = get_event_type_from_event(event)

        if event_type is None:
            continue

        parts = [EVENT_MARKERS[event_type]]

        trigger_added = False

        for start, end, raw_label in event:
            if raw_label in TRIGGER_LABEL_TO_EVENT_TYPE:
                parts.append(token_span_text(tokens, start, end))
                trigger_added = True
                break

        if not trigger_added:
            continue

        for start, end, raw_label in event:
            if raw_label not in RAW_MAIN_ARGUMENT_TO_LABEL:
                continue

            label = RAW_MAIN_ARGUMENT_TO_LABEL[raw_label]
            parts.append(MAIN_ARGUMENT_MARKERS[label])
            parts.append(token_span_text(tokens, start, end))

        event_strings.append(" ".join(parts))

    if not event_strings:
        return "none"

    return " ".join(event_strings)


def extract_gold_stage1_spans(sample):
    tokens = sample["sentence"]
    gold_spans = []

    for event in sample["event"]:
        event_type = get_event_type_from_event(event)

        for start, end, raw_label in event:
            if raw_label in TRIGGER_LABEL_TO_EVENT_TYPE:
                gold_spans.append({
                    "label": "TRIGGER",
                    "start": start,
                    "end": end,
                    "text": token_span_text(tokens, start, end),
                    "event_type": event_type,
                })

            elif raw_label in RAW_MAIN_ARGUMENT_TO_LABEL:
                gold_spans.append({
                    "label": RAW_MAIN_ARGUMENT_TO_LABEL[raw_label],
                    "start": start,
                    "end": end,
                    "text": token_span_text(tokens, start, end),
                    "event_type": event_type,
                })

    return gold_spans


def parse_stage1_output(output_text):
    if is_no_answer(output_text):
        return []

    parts = MARKER_PATTERN.split(output_text)
    events = []

    current_event = None
    current_label = None

    for part in parts:
        part = clean_piece(part)

        if not part:
            continue

        marker_key = canonical_marker(part)

        if marker_key in MARKER_TO_EVENT_TYPE:
            current_event = {
                "event_type": MARKER_TO_EVENT_TYPE[marker_key],
                "fields": {
                    "TRIGGER": [],
                    "SUBJECT": [],
                    "TREATMENT": [],
                    "EFFECT": [],
                },
            }
            events.append(current_event)
            current_label = "TRIGGER"
            continue

        if marker_key in MARKER_TO_LABEL:
            current_label = MARKER_TO_LABEL[marker_key]
            continue

        if current_event is None or current_label is None:
            continue

        if not is_no_answer(part):
            current_event["fields"].setdefault(current_label, []).append(part)

    return events


def stage1_output_to_pred_spans(output_text, tokens):
    events = parse_stage1_output(output_text)
    pred_spans = []
    seen = set()

    for event in events:
        event_type = event["event_type"]

        for label, values in event["fields"].items():
            for value in values:
                start, end = find_token_span_for_text(value, tokens)

                if start is None or end is None:
                    continue

                key = (event_type, label, start, end)

                if key in seen:
                    continue

                seen.add(key)

                pred_spans.append({
                    "label": label,
                    "start": start,
                    "end": end,
                    "text": token_span_text(tokens, start, end),
                    "generated_text": value,
                    "event_type": event_type,
                })

    return pred_spans


def extract_gold_events_for_subarguments(sample):
    tokens = sample["sentence"]
    gold_events = []

    for event_idx, event in enumerate(sample["event"]):
        event_type = get_event_type_from_event(event)
        main_arguments = []
        subarguments = []

        for start, end, raw_label in event:
            if raw_label in RAW_MAIN_ARGUMENT_TO_LABEL:
                main_arguments.append({
                    "label": RAW_MAIN_ARGUMENT_TO_LABEL[raw_label],
                    "start": start,
                    "end": end,
                    "text": token_span_text(tokens, start, end),
                    "event_type": event_type,
                })

            elif raw_label in RAW_SUBARGUMENT_TO_LABEL:
                subarguments.append({
                    "label": RAW_SUBARGUMENT_TO_LABEL[raw_label],
                    "start": start,
                    "end": end,
                    "text": token_span_text(tokens, start, end),
                    "event_type": event_type,
                })

        gold_events.append({
            "event_idx": event_idx,
            "event_type": event_type,
            "main_arguments": main_arguments,
            "subarguments": subarguments,
        })

    return gold_events


def token_span_distance(a, b):
    if a["end"] < b["start"]:
        return b["start"] - a["end"]

    if b["end"] < a["start"]:
        return a["start"] - b["end"]

    return 0


def match_pred_main_to_gold_event(pred_main_argument, gold_events):
    candidates = []

    pred_label = pred_main_argument.get("label")
    pred_event_type = pred_main_argument.get("event_type")

    for event in gold_events:
        if pred_event_type is not None and event["event_type"] != pred_event_type:
            continue

        for gold_main in event["main_arguments"]:
            if gold_main["label"] != pred_label:
                continue

            candidates.append({
                "event": event,
                "distance": token_span_distance(pred_main_argument, gold_main),
            })

    if not candidates:
        return None

    return min(candidates, key=lambda x: x["distance"])["event"]


def extract_gold_subargument_spans(sample):
    tokens = sample["sentence"]
    gold_spans = []

    for event in sample["event"]:
        event_type = get_event_type_from_event(event)

        for start, end, raw_label in event:
            if raw_label not in RAW_SUBARGUMENT_TO_LABEL:
                continue

            gold_spans.append({
                "label": RAW_SUBARGUMENT_TO_LABEL[raw_label],
                "start": start,
                "end": end,
                "text": token_span_text(tokens, start, end),
                "event_type": event_type,
            })

    return gold_spans


def split_generated_answers(text):
    if is_no_answer(text):
        return []

    parts = re.split(r"\s*;\s*", text)
    return [clean_piece(p) for p in parts if clean_piece(p) and not is_no_answer(p)]


def convert_sample_to_stage2_examples(sample, stage1_prediction_record, record_index=None):
    tokens = sample["sentence"]
    gold_events = extract_gold_events_for_subarguments(sample)
    pred_spans = stage1_prediction_record.get("pred_spans", [])

    examples = []

    predicted_main_arguments = [
        span for span in pred_spans
        if span.get("label") in SUBARGUMENT_LABELS_BY_MAIN_ARGUMENT
    ]

    for pred_idx, pred_main in enumerate(predicted_main_arguments):
        main_label = pred_main["label"]
        main_text = pred_main.get("text") or token_span_text(
            tokens,
            pred_main["start"],
            pred_main["end"],
        )

        matched_event = match_pred_main_to_gold_event(pred_main, gold_events)

        for sub_label in SUBARGUMENT_LABELS_BY_MAIN_ARGUMENT[main_label]:
            question = SUBARGUMENT_QUESTIONS[sub_label].format(
                main_argument_text=main_text
            )

            answers = []

            if matched_event is not None:
                answers = [
                    s["text"]
                    for s in matched_event["subarguments"]
                    if s["label"] == sub_label
                ]

            target_text = " ; ".join(answers) if answers else "none"

            examples.append({
                "id": f"{sample.get('id', record_index)}_predmain{pred_idx}_{sub_label}",
                "sample_id": sample.get("id", record_index),
                "record_index": record_index,
                "task": "generative_qa_stage2_subarguments",
                "input_text": build_stage2_input(question, sample),
                "target_text": target_text,
                "subargument_label": sub_label,
                "main_argument_label": main_label,
                "predicted_main_argument": pred_main,
                "matched_gold_event_idx": (
                    matched_event["event_idx"] if matched_event is not None else None
                ),
            })

    return examples


def stage2_answer_to_pred_spans(generated_answer, tokens, subargument_label, source_main_argument):
    pred_spans = []
    seen = set()

    for answer in split_generated_answers(generated_answer):
        start, end = find_token_span_for_text(answer, tokens)

        if start is None or end is None:
            continue

        key = (subargument_label, start, end, source_main_argument.get("event_type"))

        if key in seen:
            continue

        seen.add(key)

        pred_spans.append({
            "label": subargument_label,
            "start": start,
            "end": end,
            "text": token_span_text(tokens, start, end),
            "generated_text": answer,
            "event_type": source_main_argument.get("event_type"),
            "source_main_argument": {
                "label": source_main_argument.get("label"),
                "start": source_main_argument.get("start"),
                "end": source_main_argument.get("end"),
                "text": source_main_argument.get("text"),
                "event_type": source_main_argument.get("event_type"),
            },
        })

    return pred_spans

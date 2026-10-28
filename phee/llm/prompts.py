"""Prompts for zero-shot / few-shot extraction with an instruction-tuned LLM.

Following Sun et al. (2024): the best zero-shot prompt explained the schema with label
definitions, and the best few-shot setting used 5 demonstrations selected with BM25.
Differences: local open-weight models instead of ChatGPT, triggers are extracted too,
and the answer is a JSON object.
"""

import json
import re

from phee.schema import DEFINITIONS, MAIN_LABELS, SUB_LABELS, SUB_PARENT, TRIGGER

JSON_KEYS = {label: label.lower() for label in MAIN_LABELS + SUB_LABELS}
KEY_TO_LABEL = {key: label for label, key in JSON_KEYS.items()}


def _definition_lines():
    lines = [
        f"- event_type: \"ADE\" or \"PTE\". ADE: {DEFINITIONS['ADE']} PTE: {DEFINITIONS['PTE']}",
        f"- trigger: {DEFINITIONS[TRIGGER]}",
    ]
    for label in MAIN_LABELS:
        lines.append(f"- {JSON_KEYS[label]}: {DEFINITIONS[label]}")
    for label in SUB_LABELS:
        parent = JSON_KEYS[SUB_PARENT[label]]
        lines.append(f"- {JSON_KEYS[label]} (detail of the {parent}): {DEFINITIONS[label]}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You extract pharmacovigilance events from sentences of medical case reports.

An event is either an adverse drug event (ADE) or a potential therapeutic event (PTE).
For every event in the sentence give its type, its trigger and its arguments.

Fields:
{_definition_lines()}

Rules:
- Copy every text EXACTLY as it appears in the sentence (same words, same spelling). Do not paraphrase.
- Every field except event_type and trigger is a list of strings; use [] when the information is not in the sentence.
- Every sentence contains at least one event. A sentence may contain several events.
- Answer with one JSON object and nothing else, in this form:
{{"events": [{{"event_type": "ADE", "trigger": "...", {", ".join(f'"{JSON_KEYS[label]}": []' for label in MAIN_LABELS + SUB_LABELS)}}}]}}"""


def user_message(sample):
    return f"Sentence: {sample['context']}"


def gold_json(sample):
    events = []
    for event in sample["events"]:
        item = {
            "event_type": event["event_type"],
            "trigger": event["trigger"][0]["text"] if event["trigger"] else "",
        }
        for label in MAIN_LABELS:
            item[JSON_KEYS[label]] = [span["text"] for span in event["args"][label]]
        for label in SUB_LABELS:
            item[JSON_KEYS[label]] = [span["text"] for span in event["subargs"][label]]
        events.append(item)
    return json.dumps({"events": events}, ensure_ascii=False)


def messages(sample, demonstrations=()):
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    for demo in demonstrations:
        conversation.append({"role": "user", "content": user_message(demo)})
        conversation.append({"role": "assistant", "content": gold_json(demo)})
    conversation.append({"role": "user", "content": user_message(sample)})
    return conversation


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text):
    """First JSON object in the model output, or None."""
    text = _THINK.sub("", text or "")
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for position in range(start, len(text)):
            char = text[position]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:position + 1]
                    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
                        try:
                            return json.loads(attempt)
                        except json.JSONDecodeError:
                            continue
                    break
        start = text.find("{", start + 1)
    return None


def to_predictions(parsed):
    """Parsed JSON -> unified predictions; returns (predictions, valid)."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
        return [], False
    predictions = []
    for event in parsed["events"]:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type", "")).strip().upper()
        event_type = {"ADVERSE_EVENT": "ADE", "POTENTIAL_THERAPEUTIC_EVENT": "PTE"}.get(event_type, event_type)
        trigger = event.get("trigger")
        if isinstance(trigger, list):
            trigger = trigger[0] if trigger else None
        if isinstance(trigger, str) and trigger.strip() and event_type in ("ADE", "PTE"):
            predictions.append({"label": "TRIGGER", "event_type": event_type, "text": trigger})
        for key, label in KEY_TO_LABEL.items():
            values = event.get(key, [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str) and value.strip():
                    predictions.append({"label": label, "text": value})
    return predictions, True

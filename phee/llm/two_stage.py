"""Two-stage prompting with our prompt (prompt_style: two_stage; decision 27.11).

Stage 1 -- one call per sentence: the events with their type, trigger, subject, treatment
           and effect (the same definitions and rules as the one-step prompt, without the
           sub-arguments).
Stage 2 -- one call per event found in stage 1: the sentence and that event (type, trigger,
           subject, treatment, effect); the answer is the event's 12 sub-arguments.

Sun et al. (2024) tried a pipeline only zero-shot, with one question per sub-argument type:
more false positives than their best one-step prompt and about 10x the inference time.
Here stage 2 asks once per event (about 2x the calls of the one-step prompt) and uses the
same demonstrations as stage 1 (BM25, `shots` sentences with an ADE and `shots` with a PTE):
in stage 2 a demonstration is one gold event of that sentence (of the type it was retrieved
for) with its gold sub-arguments.
"""

import json

from phee.llm.prompts import (
    JSON_KEYS,
    LLM_DEFINITIONS,
    SINGLE_VALUE,
    extract_json,
    user_message,
)
from phee.schema import MAIN_LABELS, SUB_LABELS, SUB_PARENT, TRIGGER

MAIN_KEYS = [JSON_KEYS[label] for label in MAIN_LABELS]
SUB_KEYS = [JSON_KEYS[label] for label in SUB_LABELS]
_RULE_COPY = "- Copy every text EXACTLY as it appears in the sentence (same words, same spelling). Do not paraphrase."


def _stage1_fields():
    lines = [
        f"- event_type: \"ADE\" or \"PTE\". ADE: {LLM_DEFINITIONS['ADE']} PTE: {LLM_DEFINITIONS['PTE']}",
        f"- trigger: {LLM_DEFINITIONS[TRIGGER]}",
    ]
    lines += [f"- {JSON_KEYS[label]}: {LLM_DEFINITIONS[label]}" for label in MAIN_LABELS]
    return "\n".join(lines)


def _stage2_fields():
    return "\n".join(
        f"- {JSON_KEYS[label]} (detail of the {JSON_KEYS[SUB_PARENT[label]]}): {LLM_DEFINITIONS[label]}"
        for label in SUB_LABELS
    )


STAGE1_PROMPT = f"""You extract pharmacovigilance events from sentences of medical case reports.

An event is either an adverse drug event (ADE) or a potential therapeutic event (PTE).
For every event in the sentence give its type, its trigger, its subject, its treatment and its effect.

Fields:
{_stage1_fields()}

Rules:
{_RULE_COPY}
- subject is ONE string: the single fragment that describes the patients of this event; use null when the sentence does not mention them.
- treatment and effect are lists of strings; use [] when the information is not in the sentence.
- Every sentence contains at least one event. A sentence may contain several events.
- Answer with one JSON object and nothing else, in this form:
{{"events": [{{"event_type": "ADE", "trigger": "...", "subject": null, "treatment": [], "effect": []}}]}}"""


STAGE2_PROMPT = f"""You extract the details of one pharmacovigilance event from a sentence of a medical case report.

You get the sentence and one event found in it: its type (ADE: adverse drug event, PTE:
potential therapeutic event), its trigger, its subject, its treatment and its effect.
Give the details of the subject and of the treatment of THIS event only.

Fields:
{_stage2_fields()}

Rules:
{_RULE_COPY}
- Every field is a list of strings; use [] when the information is not in the sentence.
- Answer with one JSON object and nothing else, in this form:
{{{", ".join(f'"{key}": []' for key in SUB_KEYS)}}}"""


# ----------------------------------------------------------------------------- gold answers

def _trigger_text(event):
    return event["trigger"][0]["text"] if event["trigger"] else ""


def _main_fields(event):
    fields = {}
    for label in MAIN_LABELS:
        spans = sorted(event["args"][label], key=lambda span: span["start"])
        if label in SINGLE_VALUE:
            fields[JSON_KEYS[label]] = spans[0]["text"] if spans else None   # as in prompts.gold_json
        else:
            fields[JSON_KEYS[label]] = [span["text"] for span in event["args"][label]]
    return fields


def gold_event_summary(event):
    """An event as stage 1 answers it (and as stage 2 receives it)."""
    return {"event_type": event["event_type"], "trigger": _trigger_text(event), **_main_fields(event)}


def stage1_gold_json(sample):
    return json.dumps({"events": [gold_event_summary(event) for event in sample["events"]]}, ensure_ascii=False)


def stage2_gold_json(event):
    return json.dumps(
        {JSON_KEYS[label]: [span["text"] for span in event["subargs"][label]] for label in SUB_LABELS},
        ensure_ascii=False,
    )


def demonstration_event(sample, event_type=None):
    """The gold event a stage-2 demonstration shows: the first one of `event_type`
    (the type the sentence was retrieved for), else the first event."""
    for event in sample["events"]:
        if event_type is None or event["event_type"] == event_type:
            return event
    return sample["events"][0]


# ----------------------------------------------------------------------------- messages

def stage1_messages(sample, demonstrations=()):
    conversation = [{"role": "system", "content": STAGE1_PROMPT}]
    for demo in demonstrations:
        conversation.append({"role": "user", "content": user_message(demo)})
        conversation.append({"role": "assistant", "content": stage1_gold_json(demo)})
    conversation.append({"role": "user", "content": user_message(sample)})
    return conversation


def stage2_user_message(sample, event_summary):
    return f"Sentence: {sample['context']}\nEvent: {json.dumps(event_summary, ensure_ascii=False)}"


def stage2_messages(sample, event_summary, demonstrations=()):
    """demonstrations: [(demo_sample, demo_gold_event)]."""
    conversation = [{"role": "system", "content": STAGE2_PROMPT}]
    for demo, demo_event in demonstrations:
        conversation.append({"role": "user", "content": stage2_user_message(demo, gold_event_summary(demo_event))})
        conversation.append({"role": "assistant", "content": stage2_gold_json(demo_event)})
    conversation.append({"role": "user", "content": stage2_user_message(sample, event_summary)})
    return conversation


# ----------------------------------------------------------------------------- parsing

def _event_type(value):
    event_type = str(value or "").strip().upper()
    return {"ADVERSE_EVENT": "ADE", "POTENTIAL_THERAPEUTIC_EVENT": "PTE"}.get(event_type, event_type)


def _texts(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str) and v.strip()]


def parse_stage1(text):
    """Stage-1 output -> (predictions, events, valid). events: [(event_index, summary)] for
    the events of a known type, the input of stage 2 (summary: event_type, trigger,
    subject, treatment, effect as the model gave them)."""
    parsed = extract_json(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
        return [], [], False
    predictions, events = [], []
    for event_index, event in enumerate(parsed["events"]):
        if not isinstance(event, dict):
            continue
        event_type = _event_type(event.get("event_type"))
        if event_type not in ("ADE", "PTE"):
            continue
        trigger = event.get("trigger")
        if isinstance(trigger, list):
            trigger = trigger[0] if trigger else None
        if isinstance(trigger, str) and trigger.strip():
            predictions.append({"label": TRIGGER, "event_type": event_type, "text": trigger, "event": event_index})
        summary = {"event_type": event_type, "trigger": trigger if isinstance(trigger, str) else ""}
        for label in MAIN_LABELS:
            key = JSON_KEYS[label]
            texts = _texts(event.get(key))
            summary[key] = (texts[0] if texts else None) if label in SINGLE_VALUE else texts
            for value in texts:
                predictions.append({"label": label, "text": value, "event": event_index, "event_type": event_type})
        events.append((event_index, summary))
    return predictions, events, True


def parse_stage2(text, event_index, event_type):
    """Stage-2 output -> (predictions of the event's sub-arguments, valid)."""
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        return [], False
    if isinstance(parsed.get("events"), list) and parsed["events"] and isinstance(parsed["events"][0], dict):
        parsed = parsed["events"][0]          # the model answered in the stage-1 form
    predictions = []
    for label in SUB_LABELS:
        for value in _texts(parsed.get(JSON_KEYS[label])):
            predictions.append({"label": label, "text": value, "event": event_index, "event_type": event_type})
    return predictions, True

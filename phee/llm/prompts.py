"""Prompts for zero-shot / few-shot extraction with an instruction-tuned LLM.

Following Sun et al. (2024): the best zero-shot prompt explained the schema with label
definitions, and the best few-shot setting used 5 demonstrations selected with BM25.
Differences: local open-weight models instead of ChatGPT, triggers are extracted too,
and the answer is a JSON object.

Decision 24.11 (test run 22.11: precision of SUBJECT 67 vs 73 and EFFECT 65 vs 69 on dev
with 5 demonstrations, against the authors' prompt): the subject is ONE string (97% of
events have one subject fragment; the list made the model give several mentions of the
same patients) and the definitions of subject and effect follow the annotation
conventions (one most complete mention; the whole effect phrase with its modifiers,
outcomes mentioned together = one fragment). In the demonstrations an event with several
subject fragments shows the first one (the patients), so the gold JSON round trip loses
those extra fragments (test: SUBJECT recall 96 %, all arguments 99.8 F1). The shared DEFINITIONS (phee.schema, also
used by the QA variants) are not changed.
"""

import json
import re

from phee.schema import DEFINITIONS, MAIN_LABELS, SUB_LABELS, SUB_PARENT, TRIGGER

JSON_KEYS = {label: label.lower() for label in MAIN_LABELS + SUB_LABELS}
KEY_TO_LABEL = {key: label for label, key in JSON_KEYS.items()}
SINGLE_VALUE = {"SUBJECT"}       # one string (or null) instead of a list

LLM_DEFINITIONS = dict(DEFINITIONS)
LLM_DEFINITIONS["SUBJECT"] = (
    "The subject is the single phrase that names the patient or group of patients who "
    "experienced the event, including the words that describe them in that phrase (age, "
    "gender, race, number, disorder). If the patients are mentioned several times, give "
    "the most complete mention."
)
LLM_DEFINITIONS["EFFECT"] = (
    DEFINITIONS["EFFECT"]
    + " Give the whole phrase that describes it, with its modifiers (e.g. 'acute', "
    "'multiple', 'worsening of'); outcomes mentioned together in one phrase (e.g. 'rash "
    "and fever') are one fragment."
)
# Decision 27.11 (test run 24.11, 5-shot, dev: our prompt found more correct arguments than
# the authors' prompt, 293 vs 266 found by one prompt only, but gave 758 vs 576 extra ones,
# mostly time elapsed 90 vs 16 and population 52 vs 8, e.g. 'after initiation',
# 'recently', 'first', 'fourth', 'a'). The annotation: 95% of time elapsed and 98% of
# duration spans state an amount of time; population is a count ('one' is annotated 26
# times in train, so only ordinals and articles are excluded). Five demonstrations rarely
# show these rules, so the prompt states them.
LLM_DEFINITIONS["TREATMENT_TIME_ELAPSED"] = (
    "The amount of time between starting the treatment and the event, e.g. '3 weeks', "
    "'within hours', 'after 5 days'. Give it only when the sentence states an amount of "
    "time; words without an amount ('after', 'recently', 'following', 'prenatally') are "
    "not a time elapsed."
)
LLM_DEFINITIONS["TREATMENT_DURATION"] = (
    "How long the treatment lasted, as an amount of time, e.g. 'for 3 weeks', '6 months'. "
    "Give it only when the sentence states an amount of time."
)
LLM_DEFINITIONS["SUBJECT_POPULATION"] = (
    "The number of subjects when the sentence counts them, e.g. 'two', '15', 'a series of'. "
    "Ordinal words ('first', 'second', 'fourth') and articles ('a', 'the') are not a population."
)


def _definition_lines():
    lines = [
        f"- event_type: \"ADE\" or \"PTE\". ADE: {LLM_DEFINITIONS['ADE']} PTE: {LLM_DEFINITIONS['PTE']}",
        f"- trigger: {LLM_DEFINITIONS[TRIGGER]}",
    ]
    for label in MAIN_LABELS:
        lines.append(f"- {JSON_KEYS[label]}: {LLM_DEFINITIONS[label]}")
    for label in SUB_LABELS:
        parent = JSON_KEYS[SUB_PARENT[label]]
        lines.append(f"- {JSON_KEYS[label]} (detail of the {parent}): {LLM_DEFINITIONS[label]}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You extract pharmacovigilance events from sentences of medical case reports.

An event is either an adverse drug event (ADE) or a potential therapeutic event (PTE).
For every event in the sentence give its type, its trigger and its arguments.

Fields:
{_definition_lines()}

Rules:
- Copy every text EXACTLY as it appears in the sentence (same words, same spelling). Do not paraphrase.
- subject is ONE string: the single fragment that describes the patients of this event; use null when the sentence does not mention them.
- Every other field except event_type and trigger is a list of strings; use [] when the information is not in the sentence.
- Every sentence contains at least one event. A sentence may contain several events.
- Answer with one JSON object and nothing else, in this form:
{{"events": [{{"event_type": "ADE", "trigger": "...", {", ".join(f'"{JSON_KEYS[label]}": ' + ("null" if label in SINGLE_VALUE else "[]") for label in MAIN_LABELS + SUB_LABELS)}}}]}}"""


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
            texts = [span["text"] for span in event["args"][label]]
            if label in SINGLE_VALUE:
                # several fragments (3% of events, mostly a discontinuous annotation: the
                # person first, a detached disorder later, e.g. 'a woman' ... 'for lupus
                # nephritis'): the first fragment in the sentence, the mention of the patients
                spans = sorted(event["args"][label], key=lambda span: span["start"])
                item[JSON_KEYS[label]] = spans[0]["text"] if spans else None
            else:
                item[JSON_KEYS[label]] = texts
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
    for event_index, event in enumerate(parsed["events"]):
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type", "")).strip().upper()
        event_type = {"ADVERSE_EVENT": "ADE", "POTENTIAL_THERAPEUTIC_EVENT": "PTE"}.get(event_type, event_type)
        trigger = event.get("trigger")
        if isinstance(trigger, list):
            trigger = trigger[0] if trigger else None
        if isinstance(trigger, str) and trigger.strip() and event_type in ("ADE", "PTE"):
            predictions.append({"label": "TRIGGER", "event_type": event_type, "text": trigger,
                                "event": event_index})
        for key, label in KEY_TO_LABEL.items():
            values = event.get(key, [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str) and value.strip():
                    item = {"label": label, "text": value, "event": event_index}
                    if event_type in ("ADE", "PTE"):
                        item["event_type"] = event_type
                    predictions.append(item)
    return predictions, True

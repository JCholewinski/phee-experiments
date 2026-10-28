"""Target formats and output parsing for generative QA.

Stage 1 (events with trigger and main arguments), question "What are the events?":
  own      "[ADE] induced [Subject] a patient [Treatment] clozapine [Effect] eosinophilia [PTE] ..."
           (every span gets its own marker; the existing implementation)
  authors  "[Adverse event] induced [Subject] a patient [Treatment] clozapine; quetiapine [Effect] ..."
           (several spans of one role joined with "; ", as in the PHEE repository)

Stage 2 (one sub-argument type per question):
  "answer" or "answer 1 ; answer 2" or "none"

A target is a list of segments (kind, text) with kind in {"event", "role", "span",
"sep", "none"}. Targets are tokenized SEGMENT BY SEGMENT, so that constrained decoding
(phee.gqa.constrained) can follow exactly the token sequences seen in training.
"""

import re

from phee.schema import MAIN_LABELS

STAGE1_QUESTION = "What are the events?"
NONE_TEXT = "none"
SEP_TEXT = ";"

EVENT_MARKERS = {
    "own": {"ADE": "[ADE]", "PTE": "[PTE]"},
    "authors": {"ADE": "[Adverse event]", "PTE": "[Potential therapeutic event]"},
}
ROLE_MARKERS = {
    "own": {"SUBJECT": "[Subject]", "TREATMENT": "[Treatment]", "EFFECT": "[Effect]"},
    "authors": {"SUBJECT": "[Subject]", "TREATMENT": "[Treatment]", "EFFECT": "[Effect]"},
}

_MARKER_PATTERN = re.compile(
    r"(\[\s*ADE\s*\]|\[\s*PTE\s*\]|\[\s*Adverse[_\s]*event\s*\]|"
    r"\[\s*Potential[_\s]*therapeutic[_\s]*event\s*\]|"
    r"\[\s*Subject\s*\]|\[\s*Treatment\s*\]|\[\s*Effect\s*\])",
    re.IGNORECASE,
)


def build_input(question, context):
    return f"question: {question} context: {context}"


def stage1_segments(sample, target_format):
    segments = []
    for event in sample["events"]:
        if not event["trigger"]:
            continue
        segments.append(("event", EVENT_MARKERS[target_format][event["event_type"]]))
        segments.append(("span", event["trigger"][0]["text"]))
        for label in MAIN_LABELS:
            spans = [span["text"] for span in event["args"][label]]
            if not spans:
                continue
            if target_format == "own":
                for text in spans:
                    segments.append(("role", ROLE_MARKERS["own"][label]))
                    segments.append(("span", text))
            else:
                segments.append(("role", ROLE_MARKERS["authors"][label]))
                for position, text in enumerate(spans):
                    if position:
                        segments.append(("sep", SEP_TEXT))
                    segments.append(("span", text))
    return segments or [("none", NONE_TEXT)]


def stage2_segments(answers):
    if not answers:
        return [("none", NONE_TEXT)]
    segments = []
    for position, text in enumerate(answers):
        if position:
            segments.append(("sep", SEP_TEXT))
        segments.append(("span", text))
    return segments


def segments_text(segments):
    return " ".join(text for _, text in segments)


class SegmentTokenizer:
    """Tokenizes a target segment by segment so that the result equals tokenizing the
    joined text (segments are separated by one space)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        whole = self._ids("hello world")
        self.first_prefix = ""
        self.next_prefix = " "
        for first in ("", " "):
            for following in (" ", ""):
                if self._ids(first + "hello") + self._ids(following + "world") == whole:
                    self.first_prefix, self.next_prefix = first, following
                    return

    def _ids(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def segment_ids(self, text, first):
        return self._ids((self.first_prefix if first else self.next_prefix) + text)

    def encode(self, segments, max_length=None):
        ids = []
        for position, (_, text) in enumerate(segments):
            ids.extend(self.segment_ids(text, first=position == 0))
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        if max_length and len(ids) > max_length:
            ids = ids[:max_length - 1] + ([self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else [])
        return ids


def _clean(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.strip(" :;,|-")


def _is_none(text):
    return _clean(text).lower() in ("", "none", "no answer", "null", "n/a")


def _marker_meaning(marker):
    key = re.sub(r"[\s_\[\]]", "", marker.lower())
    return {
        "ade": ("event", "ADE"),
        "adverseevent": ("event", "ADE"),
        "pte": ("event", "PTE"),
        "potentialtherapeuticevent": ("event", "PTE"),
        "subject": ("role", "SUBJECT"),
        "treatment": ("role", "TREATMENT"),
        "effect": ("role", "EFFECT"),
    }.get(key)


def parse_stage1(text, split_values=False):
    """Generated stage-1 text -> list of events
    {"event_type", "trigger": [str], "args": {label: [str]}}."""
    if _is_none(text):
        return []
    events = []
    current = None
    field = None
    for piece in _MARKER_PATTERN.split(text):
        piece_clean = _clean(piece)
        if not piece_clean:
            continue
        meaning = _marker_meaning(piece_clean) if _MARKER_PATTERN.fullmatch(piece.strip()) else None
        if meaning and meaning[0] == "event":
            current = {"event_type": meaning[1], "trigger": [], "args": {label: [] for label in MAIN_LABELS}}
            events.append(current)
            field = "TRIGGER"
            continue
        if meaning and meaning[0] == "role":
            field = meaning[1]
            continue
        if current is None or field is None or _is_none(piece_clean):
            continue
        values = [_clean(v) for v in re.split(r"\s*;\s*", piece)] if split_values else [piece_clean]
        for value in values:
            if not value:
                continue
            if field == "TRIGGER":
                current["trigger"].append(value)
            else:
                current["args"][field].append(value)
    return events


def parse_stage2(text):
    if _is_none(text):
        return []
    return [_clean(value) for value in re.split(r"\s*;\s*", text) if _clean(value) and not _is_none(value)]

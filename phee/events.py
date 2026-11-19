"""Events in the predictions and the event-level metrics.

Every method saves, in each prediction item, the index of the event it belongs to
("event", per sentence) and the event type ("event_type"). Items without "event" are not
assigned to any event. From these items the events of a sentence are rebuilt, so the
saved predictions keep the event structure (needed for the event metrics and for a
later error analysis).

Event metrics (agreed for the thesis):
  * event core = (event type, drug). The drug is the normalized text of a
    Treatment.Drug sub-argument of the event; an event with several drugs gives one core
    per drug; without a drug the TREATMENT text is used, without a treatment "".
    Events of a sentence with the same core are merged (e.g. one drug, two effects).
  * arguments in the event = (event type, drug, label, text) for every main argument
    and sub-argument of the event (except the drug itself): an argument counts only if it
    is attached to an event with the right core.
Both are counted per sentence as unique items (like the argument metric) and summed
into micro precision / recall / F1. Exact match of normalized texts.
"""

from collections import defaultdict

from phee.normalize import normalize_text
from phee.schema import MAIN_LABELS, SUB_LABELS, SUB_PARENT, TRIGGER

EVENT_TYPES = ("ADE", "PTE")
_MAIN = set(MAIN_LABELS)
_SUB = set(SUB_LABELS)


def events_from_items(items):
    """Prediction items with an "event" index -> list of events
    {"event_type", "trigger": [text], "args": {label: [text]}, "subargs": {label: [text]}}."""
    by_index = defaultdict(lambda: {"event_type": None, "trigger": [], "args": defaultdict(list),
                                    "subargs": defaultdict(list)})
    for item in items or []:
        if item.get("event") is None or item.get("text") is None or not str(item["text"]).strip():
            continue
        event = by_index[item["event"]]
        if item.get("event_type") in EVENT_TYPES and (event["event_type"] is None or item["label"] == TRIGGER):
            event["event_type"] = item["event_type"]
        if item["label"] == TRIGGER:
            event["trigger"].append(item["text"])
        elif item["label"] in _MAIN:
            event["args"][item["label"]].append(item["text"])
        elif item["label"] in _SUB:
            event["subargs"][item["label"]].append(item["text"])
    return [by_index[index] for index in sorted(by_index)]


def gold_events(sample):
    """Gold events of a sample in the same format as events_from_items."""
    return [
        {
            "event_type": event["event_type"],
            "trigger": [span["text"] for span in event["trigger"]],
            "args": {label: [span["text"] for span in spans] for label, spans in event["args"].items()},
            "subargs": {label: [span["text"] for span in spans] for label, spans in event["subargs"].items()},
        }
        for event in sample["events"]
    ]


def _drugs(event):
    drugs = {normalize_text(text) for text in event["subargs"].get("TREATMENT_DRUG", [])}
    if not drugs:
        drugs = {normalize_text(text) for text in event["args"].get("TREATMENT", [])}
    return drugs or {""}


def event_items(events):
    """(core items, argument items) of one sentence."""
    cores, arguments = set(), set()
    for event in events:
        event_type = event.get("event_type")
        if event_type not in EVENT_TYPES:
            continue
        for drug in _drugs(event):
            cores.add((event_type, drug))
            for label, texts in list(event["args"].items()) + list(event["subargs"].items()):
                if label == "TREATMENT_DRUG":
                    continue
                for text in texts:
                    arguments.add((event_type, drug, label, normalize_text(text)))
    return cores, arguments


def _gap(a, b):
    """Distance in tokens between two token spans (0 if they overlap)."""
    return max(0, b["tok_start"] - a["tok_end"], a["tok_start"] - b["tok_end"])


def assign_by_nearest_trigger(items):
    """Group the flat items of sequence labeling into events (same rule for both
    versions): one event per predicted trigger; every main argument goes to the nearest
    trigger (in tokens; ties: the first); every sub-argument goes to the event of the
    smallest main argument of its parent label that contains it, otherwise to the nearest
    trigger. Without a predicted trigger the arguments stay unassigned (no event).
    Items need tok_start / tok_end; "event" and "event_type" are added in place."""
    triggers = [item for item in items if item["label"] == TRIGGER and "tok_start" in item]
    if not triggers:
        return items
    for index, trigger in enumerate(triggers):
        trigger["event"] = index

    def nearest(item):
        return min(range(len(triggers)), key=lambda index: (_gap(item, triggers[index]), index))

    mains = [item for item in items if item["label"] in _MAIN and "tok_start" in item]
    for item in mains:
        item["event"] = nearest(item)
    for item in items:
        if item["label"] not in _SUB or "tok_start" not in item:
            continue
        parents = [
            main for main in mains
            if main["label"] == SUB_PARENT[item["label"]]
            and main["tok_start"] <= item["tok_start"] and item["tok_end"] <= main["tok_end"]
        ]
        if parents:
            parent = min(parents, key=lambda main: (main["tok_end"] - main["tok_start"], _gap(main, item)))
            item["event"] = parent["event"]
        else:
            item["event"] = nearest(item)
    for item in items:
        if item.get("event") is not None:
            item["event_type"] = triggers[item["event"]]["event_type"]
    return items

"""Gold tag sequences for sequence labeling.

Two schemes:

1. Own scheme ("two_level"), the user's construction:
   * two separate models: main level (triggers with event type, SUBJECT, TREATMENT,
     EFFECT) and sub-argument level (12 sub-argument labels);
   * BIO tags;
   * a sentence with several events is used ONCE, with the spans of all events in one
     sequence (the authors duplicate the sentence per event instead);
   * spans that overlap a span of higher priority cannot be placed in one tag sequence;
     they are dropped as WHOLE spans (not cut into fragments). Priorities: trigger >
     TREATMENT > EFFECT > SUBJECT; for sub-arguments the more frequent label wins
     (schema.SUB_TAG_PRIORITY).
     Among spans of equal priority the longer one wins.

2. Authors' scheme ("authors_flat"), as in PHEE preprocess/ace.py:
   * ONE model; main argument and sub-argument flattened into a combined tag
     (e.g. I-TREATMENT+TREATMENT_DRUG);
   * IO tags (no B-);
   * the sentence is duplicated once per event in train/dev; at test time a single copy;
   * per token: the trigger tag is overridden by any argument tag; among arguments the
     last matching one in the order SUBJECT, EFFECT, TREATMENT wins; among
     sub-arguments the last matching one in the authors' order wins.
"""

from collections import Counter

from phee.schema import (
    MAIN_LABELS,
    MAIN_TAG_PRIORITY,
    SUB_LABELS,
    SUB_LABELS_BY_PARENT,
    SUB_PARENT,
    SUB_TAG_PRIORITY,
)

TRIGGER_TAG = {"ADE": "ADE_TRIGGER", "PTE": "PTE_TRIGGER"}
TAG_EVENT_TYPE = {value: key for key, value in TRIGGER_TAG.items()}

MAIN_ENTITY_TYPES = ["ADE_TRIGGER", "PTE_TRIGGER"] + MAIN_LABELS
SUB_ENTITY_TYPES = list(SUB_LABELS)


def bio_tags(entity_types):
    tags = ["O"]
    for entity in entity_types:
        tags += [f"B-{entity}", f"I-{entity}"]
    return tags


OWN_MAIN_TAGS = bio_tags(MAIN_ENTITY_TYPES)
OWN_SUB_TAGS = bio_tags(SUB_ENTITY_TYPES)


# --------------------------------------------------------------------------- own scheme

def _place_spans(num_tokens, spans):
    """Greedy whole-span placement. spans: list of (priority, tok_start, tok_end, entity)."""
    tags = ["O"] * num_tokens
    taken = [False] * num_tokens
    placed, dropped = [], []
    ordered = sorted(spans, key=lambda s: (s[0], -(s[2] - s[1]), s[1]))
    for priority, start, end, entity in ordered:
        if any(taken[start:end + 1]):
            dropped.append((start, end, entity))
            continue
        for index in range(start, end + 1):
            taken[index] = True
        tags[start] = f"B-{entity}"
        for index in range(start + 1, end + 1):
            tags[index] = f"I-{entity}"
        placed.append((start, end, entity))
    return tags, placed, dropped


def _unique(spans):
    seen = set()
    result = []
    for span in spans:
        key = (span[1], span[2], span[3])
        if key not in seen:
            seen.add(key)
            result.append(span)
    return result


def own_main_tags(sample):
    spans = []
    for event in sample["events"]:
        entity = TRIGGER_TAG[event["event_type"]]
        for span in event["trigger"]:
            spans.append((MAIN_TAG_PRIORITY["TRIGGER"], span["tok_start"], span["tok_end"], entity))
        for label, label_spans in event["args"].items():
            for span in label_spans:
                spans.append((MAIN_TAG_PRIORITY[label], span["tok_start"], span["tok_end"], label))
    tags, _, dropped = _place_spans(len(sample["tokens"]), _unique(spans))
    return tags, dropped


def own_sub_tags(sample):
    spans = []
    for event in sample["events"]:
        for label, label_spans in event["subargs"].items():
            for span in label_spans:
                spans.append((SUB_TAG_PRIORITY[label], span["tok_start"], span["tok_end"], label))
    tags, _, dropped = _place_spans(len(sample["tokens"]), _unique(spans))
    return tags, dropped


def bio_to_entities(tags):
    """BIO tags -> list of (start, end, entity) (end inclusive). An I- tag that does
    not continue an entity of the same type starts a new entity."""
    entities = []
    current = None
    for index, tag in enumerate(tags):
        if tag == "O" or tag is None:
            if current:
                entities.append(tuple(current))
            current = None
            continue
        prefix, entity = tag.split("-", 1)
        if prefix == "B" or current is None or current[2] != entity:
            if current:
                entities.append(tuple(current))
            current = [index, index, entity]
        else:
            current[1] = index
    if current:
        entities.append(tuple(current))
    return entities


# ----------------------------------------------------------------------- authors' scheme

# The authors' child order (later entries win when several match one token).
AUTHORS_CHILD_ORDER = {
    "SUBJECT": [
        "SUBJECT_RACE",
        "SUBJECT_AGE",
        "SUBJECT_GENDER",
        "SUBJECT_POPULATION",
        "SUBJECT_DISORDER",
    ],
    "TREATMENT": [
        "TREATMENT_DURATION",
        "TREATMENT_TIME_ELAPSED",
        "TREATMENT_ROUTE",
        "TREATMENT_FREQ",
        "TREATMENT_DOSAGE",
        "TREATMENT_DISORDER",
        "TREATMENT_DRUG",
    ],
    "EFFECT": [],
}
AUTHORS_MAIN_ORDER = ["SUBJECT", "EFFECT", "TREATMENT"]


def _authors_tag_inventory():
    tags = ["O", "I-ADE_TRIGGER", "I-PTE_TRIGGER"]
    for main in MAIN_LABELS:
        tags.append(f"I-{main}+")
        for sub in SUB_LABELS_BY_PARENT.get(main, []):
            tags.append(f"I-{main}+{sub}")
    for sub in SUB_LABELS:
        tags.append(f"I-+{sub}")
    return tags


AUTHORS_TAGS = _authors_tag_inventory()


def authors_event_tags(sample, event):
    """IO tags of one event, following the authors' token rule."""
    num_tokens = len(sample["tokens"])
    token_labels = [set() for _ in range(num_tokens)]
    for span in event["trigger"]:
        for index in range(span["tok_start"], span["tok_end"] + 1):
            token_labels[index].add("TRIGGER")
    for label, spans in event["args"].items():
        for span in spans:
            for index in range(span["tok_start"], span["tok_end"] + 1):
                token_labels[index].add(label)
    for label, spans in event["subargs"].items():
        for span in spans:
            for index in range(span["tok_start"], span["tok_end"] + 1):
                token_labels[index].add(label)

    tags = []
    for labels in token_labels:
        tag = "O"
        if "TRIGGER" in labels:
            tag = f"I-{TRIGGER_TAG[event['event_type']]}"
        for main in AUTHORS_MAIN_ORDER:
            match_main = main if main in labels else ""
            match_sub = ""
            for sub in AUTHORS_CHILD_ORDER[main]:
                if sub in labels:
                    match_sub = sub
            if match_main or match_sub:
                tag = f"I-{match_main}+{match_sub}"
        tags.append(tag)
    return tags


def authors_training_sequences(sample):
    """One tag sequence per event (the authors' duplication of multi-event sentences)."""
    return [authors_event_tags(sample, event) for event in sample["events"]]


def authors_tags_to_entities(tags):
    """Decode authors' IO tags into (start, end, entity) for triggers, main arguments
    and sub-arguments, as in the authors' eval_ace_pred.py: contiguous runs of the
    same main (resp. sub) type form one entity."""
    def runs(values):
        found = []
        current = None
        for index, value in enumerate(values):
            if value is None:
                if current:
                    found.append(tuple(current))
                current = None
            elif current and current[2] == value:
                current[1] = index
            else:
                if current:
                    found.append(tuple(current))
                current = [index, index, value]
        if current:
            found.append(tuple(current))
        return found

    main_values, sub_values = [], []
    for tag in tags:
        if tag in (None, "O"):
            main_values.append(None)
            sub_values.append(None)
            continue
        body = tag.split("-", 1)[1]
        if body in TAG_EVENT_TYPE:
            main_values.append(body)
            sub_values.append(None)
            continue
        main, _, sub = body.partition("+")
        main_values.append(main or None)
        sub_values.append(sub or None)
    return runs(main_values) + runs(sub_values)


def tag_statistics(samples, scheme):
    """How many gold spans cannot be represented by the tag scheme (reported once)."""
    stats = Counter()
    for sample in samples:
        if scheme == "two_level":
            _, dropped_main = own_main_tags(sample)
            _, dropped_sub = own_sub_tags(sample)
            stats["dropped_main_spans"] += len(dropped_main)
            stats["dropped_sub_spans"] += len(dropped_sub)
        stats["sentences"] += 1
    return dict(stats)


def entity_to_prediction(sample, start, end, entity):
    """(token span, entity type) -> prediction item in the unified format."""
    offsets = sample["token_offsets"]
    text = sample["context"][offsets[start][0]:offsets[end][1]]
    if entity in TAG_EVENT_TYPE:
        return {"label": "TRIGGER", "event_type": TAG_EVENT_TYPE[entity], "text": text}
    if entity in MAIN_LABELS or entity in SUB_PARENT:
        return {"label": entity, "text": text}
    return None

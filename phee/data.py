"""Conversion of the revised PHEE data to one unified format used by every method.

Input:
  * data/phee2/source/{train,dev,test}.json  revised annotations (Sun et al. 2024),
    raw text with character offsets, official document-level split.
  * data/phee_v1_tokenized/{train,dev,test}.json  the original PHEE release in the
    EEQA/ACE format. Only its tokenization (spaCy, as used by the authors) is reused,
    so that sequence labeling works on the same tokens as in the original paper.

Output: one JSON line per sentence

  {
    "id": "10082597_3",
    "context": "RESULTS: A 44-year-old man ...",
    "tokens": ["RESULTS", ":", "A", ...],
    "token_offsets": [[0, 7], [7, 8], ...],          # character offsets, end exclusive
    "is_multi_event": true,
    "events": [
      {
        "event_type": "ADE",
        "trigger": [Span],
        "args": {"SUBJECT": [Span], "TREATMENT": [Span], "EFFECT": [Span]},
        "subargs": {"SUBJECT_AGE": [Span], ...}
      }
    ]
  }

  Span = {"text", "start", "end", "tok_start", "tok_end", "exact"}
    start/end      character offsets in `context` (end exclusive)
    tok_start/end  covering token indices (end inclusive)
    exact          False when the annotation boundary falls inside a token

A discontinuous annotation (several parts) becomes several spans, as in the authors'
evaluation. Combination.Drug and the event attributes are dropped.
"""

from collections import Counter

from phee.io import read_jsonl, write_jsonl
from phee.schema import (
    MAIN_LABELS,
    SOURCE_EVENT_TYPE,
    SOURCE_MAIN_ROLE,
    SOURCE_SUB_ROLE,
    SUB_LABELS,
)

SPLITS = ["train", "dev", "test"]


def align_tokens(context, tokens):
    """Character offsets of `tokens` inside `context` (tokens appear in order)."""
    offsets = []
    position = 0
    for token in tokens:
        index = context.find(token, position)
        if index < 0 or context[position:index].strip():
            raise ValueError(f"Cannot align token {token!r} at position {position}")
        offsets.append([index, index + len(token)])
        position = index + len(token)
    if context[position:].strip():
        raise ValueError("Unaligned text after the last token")
    return offsets


def char_span_to_token_span(start, end, offsets):
    """Covering token span (inclusive) of the character span [start, end)."""
    covered = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]
    if not covered:
        return None, None, False
    tok_start, tok_end = covered[0], covered[-1]
    exact = offsets[tok_start][0] == start and offsets[tok_end][1] == end
    return tok_start, tok_end, exact


def _locate(context, text, start, stats):
    """Return (start, text) of the annotation inside `context`, repairing a wrong
    offset or a whitespace difference if needed; (None, text) when not found."""
    if context[start:start + len(text)] == text:
        return start, text
    for candidate_text in (text, " ".join(text.split())):
        candidates = []
        index = context.find(candidate_text)
        while index >= 0:
            candidates.append(index)
            index = context.find(candidate_text, index + 1)
        if candidates:
            stats["repaired_offsets"] += 1
            return min(candidates, key=lambda c: abs(c - start)), candidate_text
    stats["unresolved_offsets"] += 1
    return None, text


def _spans_from_entity(entity, context, offsets, stats):
    """All parts of an annotated entity as spans."""
    spans = []
    for texts, starts in zip(entity.get("text", []), entity.get("start", [])):
        for text, start in zip(texts, starts):
            if not text.strip():
                continue
            start, text = _locate(context, text, start, stats)
            if start is None:
                continue
            end = start + len(text)
            tok_start, tok_end, exact = char_span_to_token_span(start, end, offsets)
            if tok_start is None:
                stats["spans_without_tokens"] += 1
                continue
            if not exact:
                stats["spans_not_on_token_boundaries"] += 1
            spans.append({
                "text": text,
                "start": start,
                "end": end,
                "tok_start": tok_start,
                "tok_end": tok_end,
                "exact": exact,
            })
    return spans


def _dedupe(spans):
    seen = set()
    result = []
    for span in spans:
        key = (span["start"], span["end"])
        if key in seen:
            continue
        seen.add(key)
        result.append(span)
    return result


def convert_record(record, tokens, stats):
    context = record["context"]
    offsets = align_tokens(context, tokens)
    events = []

    for source_event in record["annotations"][0]["events"]:
        event_type = SOURCE_EVENT_TYPE.get(source_event.get("event_type"))
        if event_type is None:
            stats["skipped_events"] += 1
            continue

        trigger = _dedupe(_spans_from_entity(source_event["Trigger"], context, offsets, stats))
        args = {label: [] for label in MAIN_LABELS}
        subargs = {label: [] for label in SUB_LABELS}

        for source_role, label in SOURCE_MAIN_ROLE.items():
            entity = source_event.get(source_role)
            if not entity:
                continue
            args[label].extend(_spans_from_entity(entity, context, offsets, stats))
            for key, value in entity.items():
                if key in ("text", "start", "entity_id"):
                    continue
                sub_label = SOURCE_SUB_ROLE.get((source_role, key))
                if sub_label is None:
                    # "Combination" (Combination.Drug) is dropped on purpose.
                    stats[f"dropped_{source_role}.{key}"] += 1
                    continue
                subargs[sub_label].extend(_spans_from_entity(value, context, offsets, stats))

        events.append({
            "event_type": event_type,
            "trigger": trigger,
            "args": {label: _dedupe(spans) for label, spans in args.items()},
            "subargs": {label: _dedupe(spans) for label, spans in subargs.items()},
        })
        stats["events"] += 1

    stats["sentences"] += 1
    return {
        "id": record["id"],
        "context": context,
        "tokens": tokens,
        "token_offsets": offsets,
        "is_multi_event": len(events) > 1,
        "events": events,
    }


def convert_split(source_path, tokenized_path, output_path):
    tokens_by_id = {
        record["id"]: record["sentence"] for record in read_jsonl(tokenized_path)
    }
    stats = Counter()
    samples = []
    for record in read_jsonl(source_path):
        tokens = tokens_by_id.get(record["id"])
        if tokens is None:
            raise KeyError(f"No tokenization for sentence {record['id']}")
        samples.append(convert_record(record, tokens, stats))
    write_jsonl(samples, output_path)
    stats["multi_event_sentences"] = sum(sample["is_multi_event"] for sample in samples)
    return dict(stats)


def load_samples(path, limit=None):
    samples = read_jsonl(path)
    if limit:
        samples = samples[:limit]
    return samples


def iter_gold_spans(sample):
    """Yield (event_index, label, event_type, span) for every annotated span."""
    for event_index, event in enumerate(sample["events"]):
        for span in event["trigger"]:
            yield event_index, "TRIGGER", event["event_type"], span
        for label, spans in event["args"].items():
            for span in spans:
                yield event_index, label, event["event_type"], span
        for label, spans in event["subargs"].items():
            for span in spans:
                yield event_index, label, event["event_type"], span


def gold_predictions(sample):
    """Gold annotation in the unified prediction format (useful for tests)."""
    predictions = []
    for _, label, event_type, span in iter_gold_spans(sample):
        item = {"label": label, "text": span["text"]}
        if label == "TRIGGER":
            item["event_type"] = event_type
        predictions.append(item)
    return predictions


def token_text(sample, tok_start, tok_end):
    """Original text covered by tokens tok_start..tok_end (inclusive)."""
    offsets = sample["token_offsets"]
    return sample["context"][offsets[tok_start][0]:offsets[tok_end][1]]

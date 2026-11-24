"""Per-trigger sequence labeling (improved SL): trigger markers and per-event tags."""

from phee.data import load_samples
from phee.seq.encoding import WordEncoder
from phee.seq.tagging import (
    TRIGGER_MARKERS,
    bio_to_entities,
    event_main_tags,
    event_sub_tags,
    trigger_groups,
    trigger_tags,
)

TRAIN = "data/phee2/processed/train.jsonl"


class FakeTokenizer:
    """Every word -> one id; [CLS]=1 ... [SEP]=2; markers have their own ids."""

    vocab = {"[TRG]": 900, "[/TRG]": 901}

    def _ids(self, text):
        return [10 + sum(map(ord, text.strip())) % 800] if text.strip() else []

    def __call__(self, text, add_special_tokens=True):
        if isinstance(text, list):
            return {"input_ids": [self._ids(t) for t in text]}
        ids = self._ids(text)
        return {"input_ids": [1] + ids + [2] if add_special_tokens else ids}

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


def _sample(tokens):
    offsets, position = [], 0
    for token in tokens:
        offsets.append([position, position + len(token)])
        position += len(token) + 1
    return {"tokens": tokens, "token_offsets": offsets, "context": " ".join(tokens)}


def test_markers_are_input_tokens_only():
    encoder = WordEncoder(FakeTokenizer(), markers=TRIGGER_MARKERS)
    sample = _sample(["Isoniazid", "caused", "hepatitis", "."])
    plain = encoder.encode(sample)
    marked = encoder.encode(sample, [(1, 1)])
    # same words, tags stay aligned with the PHEE tokens
    assert marked["valid_words"] == plain["valid_words"] == [0, 1, 2, 3]
    ids = marked["input_ids"]
    assert ids.count(900) == 1 and ids.count(901) == 1
    # [CLS] Isoniazid [TRG] caused [/TRG] hepatitis . [SEP]
    assert ids.index(900) == 2 and ids.index(901) == 4
    assert [ids[p] for p in marked["word_positions"]] == [plain["input_ids"][p] for p in plain["word_positions"]]


def test_multi_event_sentence_gives_one_example_per_trigger():
    samples = [s for s in load_samples(TRAIN) if s["is_multi_event"]]
    sample = next(s for s in samples if len({e["event_type"] for e in s["events"]}) == 2)
    groups = trigger_groups(sample)
    assert len(groups) == len(sample["events"])
    for marks, events in groups:
        expected = {
            (span["tok_start"], span["tok_end"], label)
            for event in events
            for label, spans in event["args"].items()
            for span in spans
        }
        tagged = set(bio_to_entities(event_main_tags(sample, events)[0]))
        assert tagged <= expected and tagged      # only this event's arguments
        assert not any(entity.endswith("TRIGGER") for _, _, entity in tagged)
        subs = {entity for _, _, entity in bio_to_entities(event_sub_tags(sample, events)[0])}
        assert subs <= {label for event in events for label in event["subargs"]}
    trigger_entities = bio_to_entities(trigger_tags(sample)[0])
    assert len(trigger_entities) == len(groups)


def test_events_with_the_same_trigger_are_merged():
    merged = 0
    for sample in load_samples(TRAIN):
        groups = trigger_groups(sample)
        assert sum(len(events) for _, events in groups) == len(sample["events"])
        merged += sum(len(events) - 1 for _, events in groups)
    assert merged == 15


def test_predicted_trigger_is_matched_to_the_nearest_event_of_its_type():
    import torch  # noqa: F401  (the runner needs torch)
    from phee.seq.runner import match_events

    sample = {"events": [
        {"event_type": "ADE", "trigger": [{"tok_start": 1, "tok_end": 1}], "id": "a"},
        {"event_type": "PTE", "trigger": [{"tok_start": 6, "tok_end": 6}], "id": "p"},
        {"event_type": "ADE", "trigger": [{"tok_start": 9, "tok_end": 10}], "id": "b"},
    ]}
    ids = lambda events: [e["id"] for e in events]
    assert ids(match_events(sample, [(2, 2)], "ADE")) == ["a"]        # wrong word, nearest ADE
    assert ids(match_events(sample, [(7, 7)], "ADE")) == ["b"]        # PTE ignored
    assert ids(match_events(sample, [(6, 6)], "PTE")) == ["p"]
    assert match_events({"events": sample["events"][:1]}, [(3, 3)], "PTE") == []   # no event: no arguments


def test_forced_trigger_takes_the_best_trigger_word():
    import torch
    from phee.seq.runner import forced_trigger
    from phee.seq.tagging import TRIGGER_TAGS

    sample = _sample(["Isoniazid", "caused", "hepatitis", "."])
    item = {"valid_words": [0, 1, 2, 3]}
    scores = torch.zeros(4, len(TRIGGER_TAGS))
    scores[1, TRIGGER_TAGS.index("B-ADE_TRIGGER")] = 2.0
    scores[2, TRIGGER_TAGS.index("B-PTE_TRIGGER")] = 1.0
    scores[0, TRIGGER_TAGS.index("I-ADE_TRIGGER")] = 9.0               # I- tags are not used
    prediction = forced_trigger(sample, item, scores, TRIGGER_TAGS)
    assert prediction["text"] == "caused" and prediction["event_type"] == "ADE"

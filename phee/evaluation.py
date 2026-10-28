"""Text-based evaluation shared by every method.

Agreed rules (work plan, section "Dane i ewaluacja"):
  * a prediction is a pair (label, normalized text); normalization as in the authors'
    official evaluator (see phee.normalize);
  * pairs are counted per sentence as UNIQUE pairs: an argument shared by two events
    counts once, and the assignment of arguments to individual events is not checked;
  * arguments do not need a matching event type (ADE / PTE); triggers are scored with
    (classification) and without (identification) their event type;
  * EM-F1 = exact match of normalized texts; token-F1 = overlap of the bags of tokens
    of all unique spans with the same label in a sentence (as in the authors' metric).

Prediction format (one record per sentence):
    {"id": "...", "pred": [{"label": "TRIGGER", "event_type": "ADE", "text": "..."},
                           {"label": "TREATMENT", "text": "..."}, ...]}

Extra analyses:
  * results for single-event and multi-event sentences separately;
  * "trigger dependency": every gold argument is put in a 2x2 table
    (was the trigger of its event found?) x (was the argument found?), and every gold
    sub-argument in (was its parent main argument found?) x (was it found?).
"""

from collections import Counter, defaultdict

from phee.normalize import normalize_text
from phee.schema import MAIN_LABELS, SUB_LABELS, SUB_PARENT, TRIGGER

FAMILIES = [
    "trigger_identification",
    "trigger_classification",
    "event_type",
    "main_identification",
    "main_classification",
    "sub_identification",
    "sub_classification",
    "arguments_classification",
    "overall_classification",
]

_MAIN = set(MAIN_LABELS)
_SUB = set(SUB_LABELS)


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"p": precision, "r": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


class SentenceView:
    """Unique normalized items of one sentence, per family and per label."""

    def __init__(self):
        self.trigger_texts = set()
        self.triggers = set()            # (event_type, text)
        self.event_types = set()
        self.by_label = defaultdict(set)  # label -> set of texts

    @classmethod
    def from_predictions(cls, predictions):
        view = cls()
        for item in predictions or []:
            label = item.get("label")
            raw = item.get("text")
            if raw is None or not str(raw).strip():
                continue
            # As in the authors' evaluator, a text that normalizes to "" (e.g. "a")
            # is still an item.
            text = normalize_text(raw)
            if label == TRIGGER:
                view.trigger_texts.add(text)
                event_type = item.get("event_type")
                if event_type:
                    view.triggers.add((event_type, text))
                    view.event_types.add(event_type)
            elif label in _MAIN or label in _SUB:
                view.by_label[label].add(text)
        return view

    @classmethod
    def from_gold(cls, sample):
        view = cls()
        for event in sample["events"]:
            view.event_types.add(event["event_type"])
            for span in event["trigger"]:
                text = normalize_text(span["text"])
                view.trigger_texts.add(text)
                view.triggers.add((event["event_type"], text))
            for label, spans in list(event["args"].items()) + list(event["subargs"].items()):
                for span in spans:
                    view.by_label[label].add(normalize_text(span["text"]))
        return view

    def family_items(self, family):
        if family == "trigger_identification":
            return {("", text) for text in self.trigger_texts}
        if family == "trigger_classification":
            return set(self.triggers)
        if family == "event_type":
            return {(event_type, "") for event_type in self.event_types}
        if family == "main_identification":
            return {("", text) for label in MAIN_LABELS for text in self.by_label[label]}
        if family == "main_classification":
            return {(label, text) for label in MAIN_LABELS for text in self.by_label[label]}
        if family == "sub_identification":
            return {("", text) for label in SUB_LABELS for text in self.by_label[label]}
        if family == "sub_classification":
            return {(label, text) for label in SUB_LABELS for text in self.by_label[label]}
        if family == "arguments_classification":
            return {
                (label, text)
                for label in MAIN_LABELS + SUB_LABELS
                for text in self.by_label[label]
            }
        if family == "overall_classification":
            items = {("TRIGGER:" + event_type, text) for event_type, text in self.triggers}
            items |= self.family_items("arguments_classification")
            return items
        raise KeyError(family)


def _token_overlap(gold_items, pred_items):
    """Token-level counts: bags of tokens grouped by the item's label key."""
    gold_bags = defaultdict(Counter)
    pred_bags = defaultdict(Counter)
    for key, text in gold_items:
        gold_bags[key].update(text.split())
    for key, text in pred_items:
        pred_bags[key].update(text.split())
    tp = fp = fn = 0
    for key in set(gold_bags) | set(pred_bags):
        common = sum((gold_bags[key] & pred_bags[key]).values())
        tp += common
        fp += sum(pred_bags[key].values()) - common
        fn += sum(gold_bags[key].values()) - common
    return tp, fp, fn


class _Accumulator:
    def __init__(self):
        self.em = defaultdict(lambda: [0, 0, 0])
        self.token = defaultdict(lambda: [0, 0, 0])
        self.label_em = defaultdict(lambda: [0, 0, 0])
        self.label_token = defaultdict(lambda: [0, 0, 0])
        self.sentences = 0

    def add(self, gold_view, pred_view):
        self.sentences += 1
        for family in FAMILIES:
            gold_items = gold_view.family_items(family)
            pred_items = pred_view.family_items(family)
            counts = self.em[family]
            counts[0] += len(gold_items & pred_items)
            counts[1] += len(pred_items - gold_items)
            counts[2] += len(gold_items - pred_items)
            if family != "event_type":
                tp, fp, fn = _token_overlap(gold_items, pred_items)
                token_counts = self.token[family]
                token_counts[0] += tp
                token_counts[1] += fp
                token_counts[2] += fn
        for label in MAIN_LABELS + SUB_LABELS:
            gold_texts = gold_view.by_label[label]
            pred_texts = pred_view.by_label[label]
            counts = self.label_em[label]
            counts[0] += len(gold_texts & pred_texts)
            counts[1] += len(pred_texts - gold_texts)
            counts[2] += len(gold_texts - pred_texts)
            tp, fp, fn = _token_overlap(
                {(label, t) for t in gold_texts}, {(label, t) for t in pred_texts}
            )
            token_counts = self.label_token[label]
            token_counts[0] += tp
            token_counts[1] += fp
            token_counts[2] += fn
        for event_type, text in gold_view.triggers | pred_view.triggers:
            key = "TRIGGER:" + event_type
            hit = (event_type, text)
            counts = self.label_em[key]
            if hit in gold_view.triggers and hit in pred_view.triggers:
                counts[0] += 1
            elif hit in pred_view.triggers:
                counts[1] += 1
            else:
                counts[2] += 1

    def result(self):
        families = {}
        for family in FAMILIES:
            families[family] = {"em": prf(*self.em[family])}
            if family != "event_type":
                families[family]["token"] = prf(*self.token[family])
        per_label = {}
        for label, counts in self.label_em.items():
            per_label[label] = {"em": prf(*counts), "support": counts[0] + counts[2]}
            if label in self.label_token:
                per_label[label]["token"] = prf(*self.label_token[label])
        return {"sentences": self.sentences, "families": families, "per_label": per_label}


def _dependency_counts(sample, pred_view):
    """2x2 counts for arguments vs. their trigger and sub-arguments vs. their parent.

    An argument shared by several events is counted once; its trigger (parent) counts
    as found if it was found for at least one of these events."""
    main_condition = {}  # (label, text) -> trigger of some containing event found
    sub_condition = {}   # (label, text) -> parent of some containing event found
    for event in sample["events"]:
        trigger_found = any(
            normalize_text(span["text"]) in pred_view.trigger_texts for span in event["trigger"]
        )
        for label, spans in event["args"].items():
            for span in spans:
                key = (label, normalize_text(span["text"]))
                main_condition[key] = main_condition.get(key, False) or trigger_found
        for label, spans in event["subargs"].items():
            parent = SUB_PARENT[label]
            parent_found = any(
                normalize_text(parent_span["text"]) in pred_view.by_label[parent]
                for parent_span in event["args"].get(parent, [])
            )
            for span in spans:
                key = (label, normalize_text(span["text"]))
                sub_condition[key] = sub_condition.get(key, False) or parent_found

    counts = Counter()
    for (label, text), trigger_found in main_condition.items():
        found = text in pred_view.by_label[label]
        counts[
            f"main_trigger_{'found' if trigger_found else 'missed'}"
            f"_arg_{'found' if found else 'missed'}"
        ] += 1
    for (label, text), parent_found in sub_condition.items():
        found = text in pred_view.by_label[label]
        counts[
            f"sub_parent_{'found' if parent_found else 'missed'}"
            f"_arg_{'found' if found else 'missed'}"
        ] += 1
    return counts


def evaluate(samples, predictions):
    """Evaluate predictions (list of records with "id" and "pred") against gold samples."""
    pred_by_id = {record["id"]: record.get("pred", []) for record in predictions}
    overall = _Accumulator()
    subsets = {"single_event": _Accumulator(), "multi_event": _Accumulator()}
    dependency = Counter()
    missing = 0

    for sample in samples:
        if sample["id"] not in pred_by_id:
            missing += 1
        gold_view = SentenceView.from_gold(sample)
        pred_view = SentenceView.from_predictions(pred_by_id.get(sample["id"], []))
        overall.add(gold_view, pred_view)
        subset = "multi_event" if sample["is_multi_event"] else "single_event"
        subsets[subset].add(gold_view, pred_view)
        dependency.update(_dependency_counts(sample, pred_view))

    result = overall.result()
    result["missing_predictions"] = missing
    result["subsets"] = {name: acc.result() for name, acc in subsets.items()}
    result["dependency"] = dict(dependency)
    result["summary"] = summarize(result)
    return result


def summarize(result):
    families = result["families"]
    summary = {
        "trigger_identification_f1": families["trigger_identification"]["em"]["f1"],
        "trigger_classification_f1": families["trigger_classification"]["em"]["f1"],
        "event_type_f1": families["event_type"]["em"]["f1"],
        "main_identification_f1": families["main_identification"]["em"]["f1"],
        "main_classification_f1": families["main_classification"]["em"]["f1"],
        "main_classification_token_f1": families["main_classification"]["token"]["f1"],
        "sub_identification_f1": families["sub_identification"]["em"]["f1"],
        "sub_classification_f1": families["sub_classification"]["em"]["f1"],
        "sub_classification_token_f1": families["sub_classification"]["token"]["f1"],
        "arguments_classification_f1": families["arguments_classification"]["em"]["f1"],
        "overall_classification_f1": families["overall_classification"]["em"]["f1"],
    }
    for name, subset in result.get("subsets", {}).items():
        sub_families = subset["families"]
        summary[f"{name}_main_classification_f1"] = sub_families["main_classification"]["em"]["f1"]
        summary[f"{name}_sub_classification_f1"] = sub_families["sub_classification"]["em"]["f1"]
        summary[f"{name}_overall_classification_f1"] = sub_families["overall_classification"]["em"]["f1"]
    return summary


def score_families(samples, predictions, families):
    """Micro EM-F1 over the union of the given families (used for early stopping)."""
    pred_by_id = {record["id"]: record.get("pred", []) for record in predictions}
    tp = fp = fn = 0
    for sample in samples:
        gold_view = SentenceView.from_gold(sample)
        pred_view = SentenceView.from_predictions(pred_by_id.get(sample["id"], []))
        for family in families:
            gold_items = {(family, item) for item in gold_view.family_items(family)}
            pred_items = {(family, item) for item in pred_view.family_items(family)}
            tp += len(gold_items & pred_items)
            fp += len(pred_items - gold_items)
            fn += len(gold_items - pred_items)
    return prf(tp, fp, fn)

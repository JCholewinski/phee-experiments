"""The prompting setup of Sun et al. (2024), "Leveraging ChatGPT in Pharmacovigilance
Event Extraction", used as the LLM baseline (same local model, the authors' prompt).

Taken from their code (github.com/ZhaoyueSun/phee-with-chatgpt, MIT license):
  * instruction: chatgpt_few_shot/instruction_templates/explanation.json (the
    "Explanation" prompt: schema + one-line definition of every type), sent as a user
    message; query "Sentence: <sentence> Output:" (query_templates/explanation.json);
  * demonstrations (chatgpt_few_shot/prompter.py, strategy BM25-Type as in
    run_chatgpt_fewshot.py): k most similar training sentences with an adverse event
    and k with a potential therapeutic event (BM25 over whitespace tokens), each as a
    user query + an assistant answer: a JSON list of events with all 16 argument types,
    "N/A" when absent and several spans joined with "; ";
  * no triggers are asked for (the paper did not query ChatGPT for triggers);
  * parsing (chatgpt_few_shot/evaluate.py): spans "N/A", "null", "none", "not mentioned"
    are dropped, several spans are split on ";".
Differences: Combination.Drug is not annotated in our data, so "combination_drug" is
always "N/A" in the demonstrations and ignored in the answers.
"""

import json
import math
import re
from collections import Counter, OrderedDict

INSTRUCTION = (
    "Extract event information from the following sentence and return events in json format as this: "
    "[{\"event_type\": event type, \"arguments\":[{\"argument_type\": argument type, \"argument_span\":argument extraction}]}]. "
    "Event type: adverse event (an event shows the use of a drug or combination of drugs cause a harmful effect on the human patient), "
    "potential therapeutic event (an event shows the use of a drug or combination of drugs bring a potential beneficial effect on the human patient). "
    "Argument type: subject (overall description of the patients involved in the event), age (the concrete age or an age range of the subject), "
    "gender (the subject's gender), race (the subject's race or nationality), population (the number of patients receiving the treatment), "
    "subject_disorder (the subject's disorders), treatment (overall description of the therapy administered to the patients), "
    "drug (the drugs used as therapy in the event), dosage (the amount of the drug is given), route (the route of the drug administration), "
    "duration (how long the patient has been taking the medicine), frequency (the frequency of drug use), "
    "time_elapsed (the time elapsed after the drug was administered to the occurrence of the side effect), "
    "indication (the target disorder of the medicine administration), combination_drug (the drugs used in combination), "
    "effect (the side effect in the adverse event or the beneficial effect in the potential therapeutic event). "
)
QUERY = "Sentence: {sentence} Output:"

EVENT_NAME = {"ADE": "adverse event", "PTE": "potential therapeutic event"}
EVENT_FROM_NAME = {
    "adverse event": "ADE", "adverse_event": "ADE",
    "potential therapeutic event": "PTE", "potential_therapeutic_event": "PTE",
}
# authors' argument keys (in their order) -> our labels (None: not annotated here)
ARGUMENT_KEYS = OrderedDict([
    ("subject", "SUBJECT"), ("treatment", "TREATMENT"), ("effect", "EFFECT"),
    ("age", "SUBJECT_AGE"), ("gender", "SUBJECT_GENDER"), ("race", "SUBJECT_RACE"),
    ("population", "SUBJECT_POPULATION"), ("subject_disorder", "SUBJECT_DISORDER"),
    ("drug", "TREATMENT_DRUG"), ("dosage", "TREATMENT_DOSAGE"), ("route", "TREATMENT_ROUTE"),
    ("duration", "TREATMENT_DURATION"), ("frequency", "TREATMENT_FREQ"),
    ("time_elapsed", "TREATMENT_TIME_ELAPSED"), ("indication", "TREATMENT_DISORDER"),
    ("combination_drug", None),
])


def gold_answer(sample):
    events = []
    for event in sample["events"]:
        arguments = OrderedDict()
        for key, label in ARGUMENT_KEYS.items():
            spans = []
            if label is not None:
                source = event["args"] if label in event["args"] else event["subargs"]
                spans = [span["text"] for span in source.get(label, [])]
            arguments[key] = "; ".join(spans) if spans else "N/A"
        events.append({"event_type": EVENT_NAME[event["event_type"]], "arguments": arguments})
    return json.dumps(events)


def messages(sample, demonstrations=()):
    conversation = [{"role": "user", "content": INSTRUCTION}]
    for demo in demonstrations:
        conversation.append({"role": "user", "content": QUERY.format(sentence=demo["context"])})
        conversation.append({"role": "assistant", "content": gold_answer(demo)})
    conversation.append({"role": "user", "content": QUERY.format(sentence=sample["context"])})
    return conversation


class WhitespaceBM25:
    """BM25Okapi as in the rank_bm25 package used by the authors (k1 1.5, b 0.75,
    negative idf replaced by 0.25 x mean idf), over whitespace tokens."""

    def __init__(self, documents, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b = k1, b
        self.docs = [Counter(doc.split(" ")) for doc in documents]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        frequency = Counter()
        for doc in self.docs:
            frequency.update(doc.keys())
        n = len(self.docs)
        self.idf = {term: math.log(n - df + 0.5) - math.log(df + 0.5) for term, df in frequency.items()}
        mean_idf = sum(self.idf.values()) / max(1, len(self.idf))
        for term, value in self.idf.items():
            if value < 0:
                self.idf[term] = epsilon * mean_idf

    def top(self, query, k):
        terms = query.split(" ")
        scores = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for term in terms:
                tf = doc.get(term)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_length))
            scores.append(score)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


class TypeRetriever:
    """BM25-Type: k demonstrations with an ADE, then k with a PTE (no repeats)."""

    def __init__(self, train):
        self.pools = {}
        for event_type in ("ADE", "PTE"):
            pool = [s for s in train if any(e["event_type"] == event_type for e in s["events"])]
            self.pools[event_type] = (pool, WhitespaceBM25([s["context"] for s in pool]) if pool else None)

    def demonstrations(self, sample, k):
        chosen, seen = [], set()
        for event_type in ("ADE", "PTE"):
            pool, bm25 = self.pools[event_type]
            if bm25 is None:
                continue
            taken = 0
            for index in bm25.top(sample["context"], 2 * k):
                if taken >= k:
                    break
                if pool[index]["id"] in seen:
                    continue
                seen.add(pool[index]["id"])
                chosen.append(pool[index])
                taken += 1
        return chosen


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_events(text):
    """First JSON list (or object) in the answer, or None."""
    text = _THINK.sub("", text or "")
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opening, closing in (("[", "]"), ("{", "}")):
        start = text.find(opening)
        while start >= 0:
            depth, in_string, escape = 0, False, False
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
                elif char == opening:
                    depth += 1
                elif char == closing:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:position + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find(opening, start + 1)
    return None


def _skip(span):
    span = span.lower()
    return "n/a" in span or "null" in span or "none" in span or "not mentioned" in span


def _event_type(name):
    """The authors' event name ("adverse event", "potential therapeutic event") -> ADE / PTE."""
    name = str(name or "").strip().lower()
    if "therapeutic" in name or name == "pte":
        return "PTE"
    if "adverse" in name or name == "ade":
        return "ADE"
    return None


def to_predictions(parsed):
    """Authors' answer format -> unified predictions (no triggers); (predictions, valid).
    Every item keeps the index and the type of its event."""
    if isinstance(parsed, dict):
        parsed = parsed.get("events", [parsed])
    if not isinstance(parsed, list):
        return [], False
    predictions = []
    for event_index, event in enumerate(parsed):
        if not isinstance(event, dict):
            continue
        event_type = _event_type(event.get("event_type"))
        arguments = event.get("arguments", {})
        if isinstance(arguments, list):  # the format written in the instruction
            pairs = [(a.get("argument_type"), a.get("argument_span")) for a in arguments if isinstance(a, dict)]
        elif isinstance(arguments, dict):  # the format of the demonstrations
            pairs = list(arguments.items())
        else:
            continue
        for key, span in pairs:
            label = ARGUMENT_KEYS.get(str(key).strip().lower())
            if label is None or span is None:
                continue
            spans = span if isinstance(span, list) else str(span).split(";")
            for value in spans:
                value = str(value).strip()
                if value and not _skip(value):
                    item = {"label": label, "text": value, "event": event_index}
                    if event_type:
                        item["event_type"] = event_type
                    predictions.append(item)
    return predictions, True

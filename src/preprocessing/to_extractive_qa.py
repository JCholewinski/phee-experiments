# src/preprocessing/to_extractive_qa.py

from typing import Dict, List, Optional


TRIGGER_LABEL_TO_EVENT_TYPE = {
    "Adverse_event": "ADE",
    "Potential_therapeutic_event": "PTE",
}


EVENT_TYPE_TO_QUESTION = {
    "ADE": "What phrase indicates an adverse drug event?",
    "PTE": "What phrase indicates a potential therapeutic event?",
}


def tokens_to_context_and_offsets(tokens: List[str]):
    """
    Converts token list to plain text context and stores character offsets
    for every original token.

    Example:
    tokens = ["Drug", "caused", "rash"]
    context = "Drug caused rash"
    offsets = [(0, 4), (5, 11), (12, 16)]
    """

    context_parts = []
    offsets = []

    current_pos = 0

    for token in tokens:
        token = str(token)

        if context_parts:
            context_parts.append(" ")
            current_pos += 1

        start = current_pos
        context_parts.append(token)
        current_pos += len(token)
        end = current_pos

        offsets.append((start, end))

    context = "".join(context_parts)

    return context, offsets


def token_span_to_char_span(start_token: int, end_token: int, offsets):
    char_start = offsets[start_token][0]
    char_end = offsets[end_token][1]

    return char_start, char_end


def extract_trigger_from_event(event: List[List]) -> Optional[Dict]:
    """
    Finds trigger annotation inside one PHEE event.
    Returns:
    {
        "start": int,
        "end": int,
        "event_type": "ADE" / "PTE",
        "raw_label": "Adverse_event" / "Potential_therapeutic_event"
    }
    """

    for start, end, label in event:
        if label in TRIGGER_LABEL_TO_EVENT_TYPE:
            return {
                "start": start,
                "end": end,
                "event_type": TRIGGER_LABEL_TO_EVENT_TYPE[label],
                "raw_label": label,
            }

    return None


def convert_sample_to_trigger_qa(sample: Dict) -> List[Dict]:
    """
    Converts one raw PHEE sample into extractive QA examples for trigger extraction.

    One event = one QA example.
    """

    tokens = sample["sentence"]
    context, token_offsets = tokens_to_context_and_offsets(tokens)

    qa_examples = []

    for event_idx, event in enumerate(sample["event"]):
        trigger = extract_trigger_from_event(event)

        if trigger is None:
            continue

        char_start, char_end = token_span_to_char_span(
            start_token=trigger["start"],
            end_token=trigger["end"],
            offsets=token_offsets,
        )

        answer_text = context[char_start:char_end]
        question = EVENT_TYPE_TO_QUESTION[trigger["event_type"]]

        qa_examples.append(
            {
                "id": f"{sample.get('id')}_{event_idx}_{trigger['event_type']}_trigger",
                "sample_id": sample.get("id"),
                "event_idx": event_idx,
                "task": "trigger_extraction",
                "event_type": trigger["event_type"],
                "question": question,
                "context": context,
                "answers": {
                    "text": [answer_text],
                    "answer_start": [char_start],
                },
                "tokens": tokens,
                "token_offsets": token_offsets,
                "gold_trigger": {
                    "label": "TRIGGER",
                    "start": trigger["start"],
                    "end": trigger["end"],
                    "text": answer_text,
                    "event_type": trigger["event_type"],
                },
            }
        )

    return qa_examples

MAIN_ARGUMENT_LABELS = ["SUBJECT", "TREATMENT", "EFFECT"]

RAW_MAIN_ARGUMENT_TO_LABEL = {
    "Subject": "SUBJECT",
    "Treatment": "TREATMENT",
    "Effect": "EFFECT",
}

MAIN_ARGUMENT_QUESTIONS = {
    "SUBJECT": 'Who or what is involved in the event triggered by "{trigger_text}"?',
    "TREATMENT": 'What treatment is involved in the event triggered by "{trigger_text}"?',
    "EFFECT": 'What effect is described in the event triggered by "{trigger_text}"?',
}


def token_span_distance(span_a, span_b):
    if span_a["end"] < span_b["start"]:
        return span_b["start"] - span_a["end"]

    if span_b["end"] < span_a["start"]:
        return span_a["start"] - span_b["end"]

    return 0


def extract_gold_events(sample):
    tokens = sample["sentence"]
    context, token_offsets = tokens_to_context_and_offsets(tokens)

    gold_events = []

    for event_idx, event in enumerate(sample["event"]):
        event_type = None
        trigger_span = None
        main_arguments = []

        for start, end, raw_label in event:
            if raw_label in TRIGGER_LABEL_TO_EVENT_TYPE:
                event_type = TRIGGER_LABEL_TO_EVENT_TYPE[raw_label]

                char_start, char_end = token_span_to_char_span(
                    start_token=start,
                    end_token=end,
                    offsets=token_offsets,
                )

                trigger_span = {
                    "label": "TRIGGER",
                    "start": start,
                    "end": end,
                    "text": context[char_start:char_end],
                    "event_type": event_type,
                }

            if raw_label in RAW_MAIN_ARGUMENT_TO_LABEL:
                label = RAW_MAIN_ARGUMENT_TO_LABEL[raw_label]

                char_start, char_end = token_span_to_char_span(
                    start_token=start,
                    end_token=end,
                    offsets=token_offsets,
                )

                main_arguments.append(
                    {
                        "label": label,
                        "start": start,
                        "end": end,
                        "text": context[char_start:char_end],
                        "char_start": char_start,
                        "char_end": char_end,
                    }
                )

        if event_type is None or trigger_span is None:
            continue

        gold_events.append(
            {
                "event_idx": event_idx,
                "event_type": event_type,
                "trigger": trigger_span,
                "main_arguments": main_arguments,
            }
        )

    return gold_events


def match_predicted_trigger_to_gold_event(pred_trigger, gold_events):
    same_type_events = [
        event
        for event in gold_events
        if event["event_type"] == pred_trigger.get("event_type")
    ]

    if not same_type_events:
        return None

    return min(
        same_type_events,
        key=lambda event: token_span_distance(
            pred_trigger,
            event["trigger"],
        ),
    )


def convert_sample_to_main_argument_qa(sample, predicted_triggers, record_index=None):
    tokens = sample["sentence"]
    context, token_offsets = tokens_to_context_and_offsets(tokens)
    gold_events = extract_gold_events(sample)

    qa_examples = []

    for pred_trigger_idx, pred_trigger in enumerate(predicted_triggers):
        matched_event = match_predicted_trigger_to_gold_event(
            pred_trigger=pred_trigger,
            gold_events=gold_events,
        )

        trigger_text = pred_trigger.get("text")
        if not trigger_text:
            trigger_text = " ".join(
                tokens[pred_trigger["start"] : pred_trigger["end"] + 1]
            )

        for argument_label in MAIN_ARGUMENT_LABELS:
            question = MAIN_ARGUMENT_QUESTIONS[argument_label].format(
                trigger_text=trigger_text
            )

            matching_arguments = []

            if matched_event is not None:
                matching_arguments = [
                    argument
                    for argument in matched_event["main_arguments"]
                    if argument["label"] == argument_label
                ]

            if matching_arguments:
                for argument_idx, argument in enumerate(matching_arguments):
                    qa_examples.append(
                        {
                            "id": (
                                f"{sample.get('id', record_index)}_"
                                f"predtrig{pred_trigger_idx}_"
                                f"{argument_label}_{argument_idx}"
                            ),
                            "sample_id": sample.get("id", record_index),
                            "record_index": record_index,
                            "task": "main_argument_extraction",
                            "argument_label": argument_label,
                            "question": question,
                            "context": context,
                            "answers": {
                                "text": [argument["text"]],
                                "answer_start": [argument["char_start"]],
                            },
                            "tokens": tokens,
                            "token_offsets": token_offsets,
                            "predicted_trigger": pred_trigger,
                            "matched_gold_event_idx": (
                                matched_event["event_idx"]
                                if matched_event is not None
                                else None
                            ),
                        }
                    )
            else:
                qa_examples.append(
                    {
                        "id": (
                            f"{sample.get('id', record_index)}_"
                            f"predtrig{pred_trigger_idx}_"
                            f"{argument_label}_no_answer"
                        ),
                        "sample_id": sample.get("id", record_index),
                        "record_index": record_index,
                        "task": "main_argument_extraction",
                        "argument_label": argument_label,
                        "question": question,
                        "context": context,
                        "answers": {
                            "text": [],
                            "answer_start": [],
                        },
                        "tokens": tokens,
                        "token_offsets": token_offsets,
                        "predicted_trigger": pred_trigger,
                        "matched_gold_event_idx": (
                            matched_event["event_idx"]
                            if matched_event is not None
                            else None
                        ),
                    }
                )

    return qa_examples


def extract_gold_main_argument_spans(sample):
    tokens = sample["sentence"]
    context, token_offsets = tokens_to_context_and_offsets(tokens)

    gold_spans = []

    for event in sample["event"]:
        for start, end, raw_label in event:
            if raw_label not in RAW_MAIN_ARGUMENT_TO_LABEL:
                continue

            label = RAW_MAIN_ARGUMENT_TO_LABEL[raw_label]

            char_start, char_end = token_span_to_char_span(
                start_token=start,
                end_token=end,
                offsets=token_offsets,
            )

            gold_spans.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "text": context[char_start:char_end],
                } 
            )

    return gold_spans
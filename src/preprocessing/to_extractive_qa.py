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
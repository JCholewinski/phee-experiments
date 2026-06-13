# scripts/build_extractive_qa_subargument_data.py

import argparse
import json
from pathlib import Path

from src.preprocessing.to_extractive_qa import (
    RAW_MAIN_ARGUMENT_TO_LABEL,
    RAW_SUBARGUMENT_TO_LABEL,
    SUBARGUMENT_LABELS_BY_MAIN_ARGUMENT,
    SUBARGUMENT_QUESTIONS,
    token_span_to_char_span,
    tokens_to_context_and_offsets,
)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_jsonl(records, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def token_span_distance(span_a, span_b):
    if span_a["end"] < span_b["start"]:
        return span_b["start"] - span_a["end"]

    if span_b["end"] < span_a["start"]:
        return span_a["start"] - span_b["end"]

    return 0


def extract_gold_events_with_subarguments(sample):
    tokens = sample["sentence"]
    context, token_offsets = tokens_to_context_and_offsets(tokens)

    gold_events = []

    for event_idx, event in enumerate(sample["event"]):
        main_arguments = []
        subarguments = []

        for start, end, raw_label in event:
            char_start, char_end = token_span_to_char_span(
                start_token=start,
                end_token=end,
                offsets=token_offsets,
            )

            text = context[char_start:char_end]

            if raw_label in RAW_MAIN_ARGUMENT_TO_LABEL:
                main_arguments.append(
                    {
                        "label": RAW_MAIN_ARGUMENT_TO_LABEL[raw_label],
                        "start": start,
                        "end": end,
                        "text": text,
                        "char_start": char_start,
                        "char_end": char_end,
                    }
                )

            if raw_label in RAW_SUBARGUMENT_TO_LABEL:
                subarguments.append(
                    {
                        "label": RAW_SUBARGUMENT_TO_LABEL[raw_label],
                        "start": start,
                        "end": end,
                        "text": text,
                        "char_start": char_start,
                        "char_end": char_end,
                    }
                )

        gold_events.append(
            {
                "event_idx": event_idx,
                "main_arguments": main_arguments,
                "subarguments": subarguments,
            }
        )

    return gold_events


def match_predicted_main_argument_to_gold_event(predicted_main_argument, gold_events):
    predicted_label = predicted_main_argument.get("label")

    candidates = []

    for event in gold_events:
        for gold_main_argument in event["main_arguments"]:
            if gold_main_argument["label"] != predicted_label:
                continue

            distance = token_span_distance(
                predicted_main_argument,
                gold_main_argument,
            )

            candidates.append(
                {
                    "event": event,
                    "gold_main_argument": gold_main_argument,
                    "distance": distance,
                }
            )

    if not candidates:
        return None

    best_candidate = min(
        candidates,
        key=lambda candidate: candidate["distance"],
    )

    return best_candidate["event"]


def convert_sample_to_subargument_qa(sample, predicted_main_arguments, record_index=None):
    tokens = sample["sentence"]
    context, token_offsets = tokens_to_context_and_offsets(tokens)
    gold_events = extract_gold_events_with_subarguments(sample)

    qa_examples = []

    for pred_main_idx, predicted_main_argument in enumerate(predicted_main_arguments):
        main_argument_label = predicted_main_argument.get("label")

        if main_argument_label not in SUBARGUMENT_LABELS_BY_MAIN_ARGUMENT:
            continue

        main_argument_text = predicted_main_argument.get("text")

        if not main_argument_text:
            main_argument_text = " ".join(
                tokens[
                    predicted_main_argument["start"] : predicted_main_argument["end"] + 1
                ]
            )

        matched_event = match_predicted_main_argument_to_gold_event(
            predicted_main_argument=predicted_main_argument,
            gold_events=gold_events,
        )

        for subargument_label in SUBARGUMENT_LABELS_BY_MAIN_ARGUMENT[
            main_argument_label
        ]:
            question = SUBARGUMENT_QUESTIONS[subargument_label].format(
                main_argument_text=main_argument_text
            )

            matching_subarguments = []

            if matched_event is not None:
                matching_subarguments = [
                    subargument
                    for subargument in matched_event["subarguments"]
                    if subargument["label"] == subargument_label
                ]

            if matching_subarguments:
                for subargument_idx, subargument in enumerate(matching_subarguments):
                    qa_examples.append(
                        {
                            "id": (
                                f"{sample.get('id', record_index)}_"
                                f"predmain{pred_main_idx}_"
                                f"{subargument_label}_{subargument_idx}"
                            ),
                            "sample_id": sample.get("id", record_index),
                            "record_index": record_index,
                            "task": "subargument_extraction",
                            "main_argument_label": main_argument_label,
                            "subargument_label": subargument_label,
                            "question": question,
                            "context": context,
                            "answers": {
                                "text": [subargument["text"]],
                                "answer_start": [subargument["char_start"]],
                            },
                            "tokens": tokens,
                            "token_offsets": token_offsets,
                            "predicted_main_argument": predicted_main_argument,
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
                            f"predmain{pred_main_idx}_"
                            f"{subargument_label}_no_answer"
                        ),
                        "sample_id": sample.get("id", record_index),
                        "record_index": record_index,
                        "task": "subargument_extraction",
                        "main_argument_label": main_argument_label,
                        "subargument_label": subargument_label,
                        "question": question,
                        "context": context,
                        "answers": {
                            "text": [],
                            "answer_start": [],
                        },
                        "tokens": tokens,
                        "token_offsets": token_offsets,
                        "predicted_main_argument": predicted_main_argument,
                        "matched_gold_event_idx": (
                            matched_event["event_idx"]
                            if matched_event is not None
                            else None
                        ),
                    }
                )

    return qa_examples


def build_split(raw_path, main_argument_predictions_path, output_path):
    raw_samples = load_jsonl(raw_path)
    main_argument_predictions = load_jsonl(main_argument_predictions_path)

    if len(raw_samples) != len(main_argument_predictions):
        raise ValueError(
            f"Length mismatch: raw={len(raw_samples)}, "
            f"main_argument_predictions={len(main_argument_predictions)}"
        )

    qa_examples = []

    for record_index, (sample, main_argument_record) in enumerate(
        zip(raw_samples, main_argument_predictions)
    ):
        predicted_main_arguments = main_argument_record.get("pred_spans", [])

        qa_examples.extend(
            convert_sample_to_subargument_qa(
                sample=sample,
                predicted_main_arguments=predicted_main_arguments,
                record_index=record_index,
            )
        )

    save_jsonl(qa_examples, output_path)

    no_answer = sum(
        1
        for example in qa_examples
        if not example["answers"]["answer_start"]
    )

    positive = len(qa_examples) - no_answer

    print("=" * 80)
    print(f"Saved to: {output_path}")
    print(f"Raw samples: {len(raw_samples)}")
    print(f"Main argument prediction records: {len(main_argument_predictions)}")
    print(f"Subargument QA examples: {len(qa_examples)}")
    print(f"Positive examples: {positive}")
    print(f"No-answer examples: {no_answer}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_raw_path", default="data/raw/train_main.json")
    parser.add_argument("--val_raw_path", default="data/raw/val_main.json")
    parser.add_argument("--test_raw_path", default="data/raw/test_main.json")

    parser.add_argument(
        "--train_main_argument_predictions_path",
        default="outputs/predictions/extractive_qa_main_arguments_train.jsonl",
    )
    parser.add_argument(
        "--val_main_argument_predictions_path",
        default="outputs/predictions/extractive_qa_main_arguments_val.jsonl",
    )
    parser.add_argument(
        "--test_main_argument_predictions_path",
        default="outputs/predictions/extractive_qa_main_arguments_test.jsonl",
    )

    parser.add_argument(
        "--output_dir",
        default="data/processed/extractive_qa_subarguments",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_split(
        raw_path=args.train_raw_path,
        main_argument_predictions_path=args.train_main_argument_predictions_path,
        output_path=output_dir / "train.jsonl",
    )

    build_split(
        raw_path=args.val_raw_path,
        main_argument_predictions_path=args.val_main_argument_predictions_path,
        output_path=output_dir / "val.jsonl",
    )

    build_split(
        raw_path=args.test_raw_path,
        main_argument_predictions_path=args.test_main_argument_predictions_path,
        output_path=output_dir / "test.jsonl",
    )


if __name__ == "__main__":
    main()
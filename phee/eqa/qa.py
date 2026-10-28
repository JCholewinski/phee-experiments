"""Extractive QA model utilities: features, batching and n-best answer selection.

An example is a dict:
    {"question": str, "context": str, "answer_start": int | None, "answer_text": str | None, ...}
(answer_start None = no answer). Several gold answers to one question are represented
by several training examples, as in the existing implementation.
"""

import inspect

import torch

from phee.utils import autocast_context, get_device


def accepts_token_type_ids(model):
    return "token_type_ids" in inspect.signature(model.forward).parameters


def _tokenize(tokenizer, questions, contexts, max_length, stride):
    return tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )


def _cls_index(input_ids, tokenizer):
    if tokenizer.cls_token_id is not None and tokenizer.cls_token_id in input_ids:
        return input_ids.index(tokenizer.cls_token_id)
    return 0


def train_features(examples, tokenizer, max_length=384, stride=128, use_token_type_ids=True):
    encoded = _tokenize(
        tokenizer,
        [example["question"] for example in examples],
        [example["context"] for example in examples],
        max_length,
        stride,
    )
    features = []
    for i in range(len(encoded["input_ids"])):
        example = examples[encoded["overflow_to_sample_mapping"][i]]
        input_ids = encoded["input_ids"][i]
        offsets = encoded["offset_mapping"][i]
        sequence_ids = encoded.sequence_ids(i)
        cls_index = _cls_index(input_ids, tokenizer)
        start_position = end_position = cls_index

        if example.get("answer_start") is not None:
            answer_start = example["answer_start"]
            answer_end = answer_start + len(example["answer_text"])
            context_tokens = [k for k, s in enumerate(sequence_ids) if s == 1]
            if context_tokens:
                first, last = context_tokens[0], context_tokens[-1]
                if offsets[first][0] <= answer_start and offsets[last][1] >= answer_end:
                    start = first
                    while start <= last and offsets[start][1] <= answer_start:
                        start += 1
                    end = last
                    while end >= first and offsets[end][0] >= answer_end:
                        end -= 1
                    if start <= end:
                        start_position, end_position = start, end

        feature = {
            "input_ids": input_ids,
            "start_positions": start_position,
            "end_positions": end_position,
        }
        if use_token_type_ids and "token_type_ids" in encoded:
            feature["token_type_ids"] = encoded["token_type_ids"][i]
        features.append(feature)
    return features


def make_collate(pad_token_id):
    def collate(features):
        batch_size = len(features)
        length = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full((batch_size, length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, length), dtype=torch.long)
        has_types = "token_type_ids" in features[0]
        token_type_ids = torch.zeros((batch_size, length), dtype=torch.long) if has_types else None
        for row, feature in enumerate(features):
            n = len(feature["input_ids"])
            input_ids[row, :n] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row, :n] = 1
            if has_types:
                token_type_ids[row, :n] = torch.tensor(feature["token_type_ids"], dtype=torch.long)
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if has_types:
            batch["token_type_ids"] = token_type_ids
        if "start_positions" in features[0]:
            batch["start_positions"] = torch.tensor([f["start_positions"] for f in features], dtype=torch.long)
            batch["end_positions"] = torch.tensor([f["end_positions"] for f in features], dtype=torch.long)
        return batch

    return collate


def predict_answers(model, tokenizer, examples, settings, max_answer_length=30,
                    n_best_size=20, max_spans=1, use_null=True, null_threshold=0.0,
                    max_length=384, stride=128):
    """Answers for every example: list (per example) of
    {"text", "start", "end", "score"} with non-overlapping character spans.

    use_null  answer only if the span score exceeds the "no answer" ([CLS]) score
              plus `null_threshold`; otherwise the top `max_spans` spans are returned.
    """
    if not examples:
        return []
    device = get_device()
    use_types = accepts_token_type_ids(model)
    encoded = _tokenize(
        tokenizer,
        [example["question"] for example in examples],
        [example["context"] for example in examples],
        max_length,
        stride,
    )
    collate = make_collate(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    features = []
    for i in range(len(encoded["input_ids"])):
        feature = {"input_ids": encoded["input_ids"][i]}
        if use_types and "token_type_ids" in encoded:
            feature["token_type_ids"] = encoded["token_type_ids"][i]
        features.append(feature)

    start_logits, end_logits = [], []
    batch_size = settings.get("eval_batch_size", 64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = collate(features[start:start + batch_size])
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(settings.get("precision", "bf16")):
                outputs = model(**batch)
            for row in range(outputs.start_logits.size(0)):
                n = len(features[start + row]["input_ids"])
                start_logits.append(outputs.start_logits[row, :n].float().cpu())
                end_logits.append(outputs.end_logits[row, :n].float().cpu())

    candidates = [[] for _ in examples]
    null_scores = [None] * len(examples)
    for i in range(len(features)):
        example_index = encoded["overflow_to_sample_mapping"][i]
        context = examples[example_index]["context"]
        offsets = encoded["offset_mapping"][i]
        sequence_ids = encoded.sequence_ids(i)
        input_ids = encoded["input_ids"][i]
        s_logits, e_logits = start_logits[i], end_logits[i]
        cls_index = _cls_index(input_ids, tokenizer)
        null_score = float(s_logits[cls_index] + e_logits[cls_index])
        if null_scores[example_index] is None or null_score < null_scores[example_index]:
            null_scores[example_index] = null_score

        context_positions = [
            k for k, s in enumerate(sequence_ids)
            if s == 1 and offsets[k][1] > offsets[k][0]
        ]
        if not context_positions:
            continue
        top_starts = sorted(context_positions, key=lambda k: float(s_logits[k]), reverse=True)[:n_best_size]
        top_ends = sorted(context_positions, key=lambda k: float(e_logits[k]), reverse=True)[:n_best_size]
        for s in top_starts:
            for e in top_ends:
                if e < s or e - s + 1 > max_answer_length:
                    continue
                char_start, char_end = offsets[s][0], offsets[e][1]
                # byte-level BPE offsets may include the preceding space
                while char_start < char_end and context[char_start].isspace():
                    char_start += 1
                while char_end > char_start and context[char_end - 1].isspace():
                    char_end -= 1
                if char_end <= char_start:
                    continue
                candidates[example_index].append({
                    "text": context[char_start:char_end],
                    "start": char_start,
                    "end": char_end,
                    "score": float(s_logits[s] + e_logits[e]),
                })

    answers = []
    for example_index, example_candidates in enumerate(candidates):
        example_candidates.sort(key=lambda c: c["score"], reverse=True)
        selected = []
        for candidate in example_candidates:
            if use_null and candidate["score"] <= null_scores[example_index] + null_threshold:
                break
            if any(candidate["start"] < other["end"] and other["start"] < candidate["end"] for other in selected):
                continue
            selected.append(candidate)
            if len(selected) >= max_spans:
                break
        answers.append(selected)
    return answers

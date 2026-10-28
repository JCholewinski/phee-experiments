"""Word-level encoding for the tagger: every PHEE token is tokenized separately (with the
original spacing in front of it, so sub-words look like in running text) and the model
reads the first sub-word of every word."""

import torch


class WordEncoder:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        probe = tokenizer("medication", add_special_tokens=False)["input_ids"]
        built = tokenizer.build_inputs_with_special_tokens(probe)
        self.prefix = built.index(probe[0])
        self.num_special = len(built) - len(probe)

    def encode(self, sample):
        tokens = sample["tokens"]
        offsets = sample["token_offsets"]
        context = sample["context"]
        texts = []
        for index, token in enumerate(tokens):
            gap = context[offsets[index - 1][1]:offsets[index][0]] if index else ""
            texts.append((" " if gap and not gap.strip() else "") + token if token.strip() else "")
        pieces = self.tokenizer(texts, add_special_tokens=False)["input_ids"] if texts else []

        budget = self.max_length - self.num_special
        ids, word_positions, valid_words = [], [], []
        for index, word_ids in enumerate(pieces):
            if not word_ids or not tokens[index].strip():
                continue
            if len(ids) + len(word_ids) > budget:
                break
            word_positions.append(self.prefix + len(ids))
            valid_words.append(index)
            ids.extend(word_ids)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(ids)
        return {
            "input_ids": input_ids,
            "word_positions": word_positions,
            "valid_words": valid_words,
        }


def make_collate(pad_token_id):
    def collate(items):
        batch_size = len(items)
        max_tokens = max(len(item["input_ids"]) for item in items)
        max_words = max(1, max(len(item["word_positions"]) for item in items))
        input_ids = torch.full((batch_size, max_tokens), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.long)
        word_positions = torch.zeros((batch_size, max_words), dtype=torch.long)
        word_mask = torch.zeros((batch_size, max_words), dtype=torch.bool)
        has_labels = "labels" in items[0]
        labels = torch.full((batch_size, max_words), -100, dtype=torch.long)
        for row, item in enumerate(items):
            n_tokens = len(item["input_ids"])
            n_words = len(item["word_positions"])
            input_ids[row, :n_tokens] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[row, :n_tokens] = 1
            if n_words:
                word_positions[row, :n_words] = torch.tensor(item["word_positions"], dtype=torch.long)
                word_mask[row, :n_words] = True
                if has_labels:
                    labels[row, :n_words] = torch.tensor(item["labels"], dtype=torch.long)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_positions": word_positions,
            "word_mask": word_mask,
        }
        if has_labels:
            batch["labels"] = labels
        return batch

    return collate

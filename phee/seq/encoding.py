"""Word-level encoding for the tagger: every PHEE token is tokenized separately (with the
original spacing in front of it, so sub-words look like in running text) and the model
reads the first sub-word of every word."""


class WordEncoder:
    def __init__(self, tokenizer, max_length=512, markers=None):
        """markers: (open, close) marker tokens (already added to the tokenizer) put
        around the marked spans of encode(sample, marks); they are input tokens only,
        not words, so the tags stay aligned with the PHEE tokens."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.marker_ids = None
        if markers:
            self.marker_ids = tuple(tokenizer.convert_tokens_to_ids(marker) for marker in markers)
        # The special tokens around a sequence ([CLS] ... [SEP]) are taken from a normal
        # tokenizer call: build_inputs_with_special_tokens() of a generic fast tokenizer
        # (e.g. ModernBERT) does not add them.
        probe = tokenizer("medication", add_special_tokens=False)["input_ids"]
        full = tokenizer("medication")["input_ids"]
        start = next(
            (i for i in range(len(full) - len(probe) + 1) if full[i:i + len(probe)] == probe), None
        )
        if start is None:
            raise ValueError("Cannot locate the special-token template of the tokenizer")
        self.prefix_ids = full[:start]
        self.suffix_ids = full[start + len(probe):]
        self.prefix = len(self.prefix_ids)
        self.num_special = len(self.prefix_ids) + len(self.suffix_ids)

    def encode(self, sample, marks=()):
        tokens = sample["tokens"]
        offsets = sample["token_offsets"]
        context = sample["context"]
        texts = []
        for index, token in enumerate(tokens):
            gap = context[offsets[index - 1][1]:offsets[index][0]] if index else ""
            texts.append((" " if gap and not gap.strip() else "") + token if token.strip() else "")
        pieces = self.tokenizer(texts, add_special_tokens=False)["input_ids"] if texts else []

        if marks and self.marker_ids is None:
            raise ValueError("WordEncoder was created without markers")
        opens = {start for start, _ in marks}
        closes = {end for _, end in marks}
        budget = self.max_length - self.num_special - 2 * len(marks)
        ids, word_positions, valid_words = [], [], []
        for index, word_ids in enumerate(pieces):
            if index in opens:
                ids.append(self.marker_ids[0])
            if word_ids and tokens[index].strip():
                if len(ids) + len(word_ids) > budget:
                    break
                word_positions.append(self.prefix + len(ids))
                valid_words.append(index)
                ids.extend(word_ids)
            if index in closes:
                ids.append(self.marker_ids[1])
        input_ids = self.prefix_ids + ids + self.suffix_ids
        return {
            "input_ids": input_ids,
            "word_positions": word_positions,
            "valid_words": valid_words,
        }


def make_collate(pad_token_id):
    import torch

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

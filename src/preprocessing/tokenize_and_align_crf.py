def tokenize_and_align_crf(example, tokenizer, label2id):
    """
    Tokenization/label alignment for CRF.

    Keeps labels compatible with normal BIO alignment, but adds `crf_mask`.

    crf_mask:
        1 = position used by CRF
        0 = position ignored by CRF

    We keep [CLS] active as O because torchcrf expects the first timestep
    to be active for every sequence.
    """

    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
    )

    word_ids = tokenized.word_ids()

    labels = []
    crf_mask = []
    previous_word_id = None

    for i, word_id in enumerate(word_ids):
        # [CLS], [SEP], [PAD], etc.
        if word_id is None:
            labels.append(-100)
            crf_mask.append(0)

        # First subword of an original word
        elif word_id != previous_word_id:
            labels.append(label2id[example["labels"][word_id]])
            crf_mask.append(1)

        # Continuation subword
        else:
            labels.append(-100)
            crf_mask.append(0)

        previous_word_id = word_id

    # torchcrf expects the first timestep to be active.
    # Use [CLS] as an artificial O label.
    labels[0] = label2id["O"]
    crf_mask[0] = 1

    tokenized["labels"] = labels
    tokenized["crf_mask"] = crf_mask

    return tokenized
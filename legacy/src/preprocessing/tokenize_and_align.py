def tokenize_and_align(example, tokenizer, label2id):

    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length"
    )

    word_ids = tokenized.word_ids()

    # ignorowanie pozostalych tokenow ze slowa do liczenia lossu
    labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)

        elif word_id != previous_word_id:
            labels.append(label2id[example["labels"][word_id]])

        else:
            labels.append(-100)

        previous_word_id = word_id

    tokenized["labels"] = labels

    return tokenized
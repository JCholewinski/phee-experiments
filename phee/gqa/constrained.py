"""Constrained decoding for generative QA (work plan: own improvement of the method).

The decoder may only produce outputs that follow the target grammar, and every span it
writes must be a fragment of the input sentence (between word boundaries of the
sentence). Implemented with `prefix_allowed_tokens_fn` of `model.generate()`:

  * markers / separator / "none" and all candidate spans of the sentence are stored
    in prefix tries of token ids, tokenized exactly as the training targets
    (phee.gqa.format.SegmentTokenizer);
  * after every generated token the set of possible parser states is updated; the
    allowed next tokens are the children of the trie nodes in these states.

The technique follows trie-constrained decoding used in generative information
extraction (e.g. Text2Event, Lu et al. 2021; GENRE, De Cao et al. 2021). The authors of
PHEE generated freely.

Grammars:
  stage1  start: event | none ; event -> span ; role -> span ; sep -> span ;
          span -> event | role | sep (authors format only) | END ; none -> END
  stage2  start: span | none ; span -> sep | END ; sep -> span ; none -> END
"""

END = -1

GRAMMARS = {
    "stage1_own": {
        "start": ["event", "none"],
        "event": ["span"],
        "role": ["span"],
        "span": ["event", "role", "END"],
        "sep": ["span"],
        "none": ["END"],
    },
    "stage1_authors": {
        "start": ["event", "none"],
        "event": ["span"],
        "role": ["span"],
        "span": ["event", "role", "sep", "END"],
        "sep": ["span"],
        "none": ["END"],
    },
    "stage2": {
        "start": ["span", "none"],
        "span": ["sep", "END"],
        "sep": ["span"],
        "none": ["END"],
    },
}


def _insert(trie, ids):
    node = trie
    for token in ids:
        node = node.setdefault(token, {})
    node[END] = True


def candidate_spans(sample, max_span_words=80):
    tokens = sample["tokens"]
    offsets = sample["token_offsets"]
    context = sample["context"]
    spans = []
    for i in range(len(tokens)):
        if not tokens[i].strip():
            continue
        for j in range(i, min(len(tokens), i + max_span_words)):
            if not tokens[j].strip():
                continue
            spans.append(context[offsets[i][0]:offsets[j][1]])
    return spans


class ConstraintBuilder:
    def __init__(self, segment_tokenizer, symbols, grammar, max_span_words=80):
        """symbols: {"event": [texts], "role": [texts], "sep": [texts], "none": [texts]}"""
        self.segment_tokenizer = segment_tokenizer
        self.grammar = GRAMMARS[grammar]
        self.max_span_words = max_span_words
        tokenizer = segment_tokenizer.tokenizer
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        self.symbol_tries = {}
        for kind, texts in symbols.items():
            for first in (True, False):
                trie = {}
                for text in texts:
                    _insert(trie, segment_tokenizer.segment_ids(text, first))
                self.symbol_tries[(kind, first)] = trie
        self._span_cache = {}

    def span_tries(self, sample):
        key = sample["id"]
        if key not in self._span_cache:
            texts = candidate_spans(sample, self.max_span_words)
            tries = {}
            for first in (True, False):
                prefix = self.segment_tokenizer.first_prefix if first else self.segment_tokenizer.next_prefix
                encoded = self.segment_tokenizer.tokenizer(
                    [prefix + text for text in texts], add_special_tokens=False
                )["input_ids"] if texts else []
                trie = {}
                for ids in encoded:
                    if ids:
                        _insert(trie, ids)
                tries[first] = trie
            self._span_cache[key] = tries
        return self._span_cache[key]

    def clear(self):
        self._span_cache = {}

    def prefix_allowed_tokens_fn(self, samples):
        """Function for model.generate(); samples[i] is the sentence of batch row i."""
        tries_per_row = [self.span_tries(sample) for sample in samples]
        caches = [dict() for _ in samples]

        def trie_for(row, kind, first):
            if kind == "span":
                return tries_per_row[row][first]
            return self.symbol_tries[(kind, first)]

        def start_states(row, after, first):
            states = []
            for kind in self.grammar[after]:
                if kind == "END":
                    states.append(("end", None, None))
                else:
                    states.append(("in", kind, trie_for(row, kind, first)))
            return states

        def advance(row, states, token):
            result = []
            seen = set()
            for state in states:
                if state[0] != "in":
                    continue
                _, kind, node = state
                child = node.get(token)
                if child is None:
                    continue
                if any(key != END for key in child):
                    key = ("in", kind, id(child))
                    if key not in seen:
                        seen.add(key)
                        result.append(("in", kind, child))
                if END in child:
                    for next_state in start_states(row, kind, first=False):
                        key = (next_state[0], next_state[1], id(next_state[2]))
                        if key not in seen:
                            seen.add(key)
                            result.append(next_state)
            return result

        def states_for(row, prefix):
            cache = caches[row]
            if prefix in cache:
                return cache[prefix]
            if not prefix:
                states = start_states(row, "start", first=True)
            else:
                states = advance(row, states_for(row, prefix[:-1]), prefix[-1])
            cache[prefix] = states
            return states

        def allowed(batch_id, input_ids):
            prefix = tuple(input_ids.tolist()[1:])  # drop the decoder start token
            if prefix and prefix[-1] in (self.eos, self.pad):
                return [self.pad if self.pad is not None else self.eos]
            tokens = set()
            for state in states_for(batch_id, prefix):
                if state[0] == "end":
                    tokens.add(self.eos)
                else:
                    tokens.update(key for key in state[2] if key != END)
            if not tokens:
                tokens.add(self.eos)
            return sorted(tokens)

        return allowed

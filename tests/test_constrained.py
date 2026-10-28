"""Constrained decoding accepts the gold targets (no model needed): a T5-like and a
Gemma-like SentencePiece tokenizer are trained on the data in a temporary directory.

    python -m pytest tests/test_constrained.py -q
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phee.data import load_samples  # noqa: E402
from phee.gqa.constrained import ConstraintBuilder  # noqa: E402
from phee.gqa.format import (  # noqa: E402
    EVENT_MARKERS,
    NONE_TEXT,
    ROLE_MARKERS,
    SEP_TEXT,
    SegmentTokenizer,
    segments_text,
    stage1_segments,
    stage2_segments,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "phee2" / "processed"


class _Ids(list):
    def tolist(self):
        return list(self)


def _tokenizers(directory):
    import sentencepiece as spm
    from transformers import AutoTokenizer, GemmaTokenizer, T5Tokenizer

    texts = [s["context"] for s in load_samples(DATA / "train.jsonl")]
    texts.append("question: What are the events? [ADE] [PTE] [Adverse event] [Potential therapeutic event] "
                 "[Subject] [Treatment] [Effect] none ;")
    corpus = Path(directory) / "corpus.txt"
    corpus.write_text("\n".join(texts), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(Path(directory) / "t5"), vocab_size=1200,
                                   model_type="unigram", pad_id=0, eos_id=1, unk_id=2, bos_id=-1,
                                   minloglevel=2)
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(Path(directory) / "gemma"), vocab_size=1200,
                                   model_type="bpe", pad_id=0, eos_id=1, bos_id=2, unk_id=3, byte_fallback=True,
                                   add_dummy_prefix=False, minloglevel=2)
    T5Tokenizer(vocab_file=str(Path(directory) / "t5.model"), extra_ids=0, legacy=False).save_pretrained(
        Path(directory) / "t5tok")
    GemmaTokenizer(vocab_file=str(Path(directory) / "gemma.model"), pad_token="<pad>", eos_token="</s>",
                   bos_token="<s>", unk_token="<unk>").save_pretrained(Path(directory) / "gemmatok")
    return {
        "t5": (AutoTokenizer.from_pretrained(Path(directory) / "t5tok"), 0),        # decoder start = pad
        "gemma": (AutoTokenizer.from_pretrained(Path(directory) / "gemmatok"), 2),  # decoder start = bos
    }


def _accepted(function, start_token, ids):
    prefix = [start_token]
    for token in ids:
        if token not in function(0, _Ids(prefix)):
            return False
        prefix.append(token)
    return True


def test_gold_targets_pass_the_constraints():
    samples = load_samples(DATA / "test.jsonl")[:150]
    with tempfile.TemporaryDirectory() as directory:
        for name, (tokenizer, start_token) in _tokenizers(directory).items():
            segment_tokenizer = SegmentTokenizer(tokenizer)
            for target_format in ("own", "authors"):
                symbols = {
                    "event": list(EVENT_MARKERS[target_format].values()),
                    "role": list(ROLE_MARKERS[target_format].values()),
                    "sep": [SEP_TEXT],
                    "none": [NONE_TEXT],
                }
                grammar = "stage1_authors" if target_format == "authors" else "stage1_own"
                builder = ConstraintBuilder(segment_tokenizer, symbols, grammar)
                for sample in samples:
                    segments = stage1_segments(sample, target_format)
                    ids = segment_tokenizer.encode(segments)
                    full = tokenizer(segments_text(segments), add_special_tokens=False)["input_ids"]
                    assert ids == full + [tokenizer.eos_token_id], (name, sample["id"])
                    exact = all(
                        span["exact"]
                        for event in sample["events"]
                        for span in event["trigger"] + [s for spans in event["args"].values() for s in spans]
                    )
                    if exact:
                        function = builder.prefix_allowed_tokens_fn([sample])
                        assert _accepted(function, start_token, ids), (name, target_format, sample["id"])

            builder = ConstraintBuilder(segment_tokenizer, {"sep": [SEP_TEXT], "none": [NONE_TEXT]}, "stage2")
            for sample in samples:
                function = builder.prefix_allowed_tokens_fn([sample])
                for event in sample["events"]:
                    for spans in event["subargs"].values():
                        if all(span["exact"] for span in spans):
                            ids = segment_tokenizer.encode(stage2_segments([span["text"] for span in spans]))
                            assert _accepted(function, start_token, ids), (name, sample["id"])


if __name__ == "__main__":
    test_gold_targets_pass_the_constraints()
    print("ok   test_gold_targets_pass_the_constraints")

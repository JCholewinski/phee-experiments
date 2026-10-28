"""Build tiny, randomly initialised models with locally trained tokenizers for smoke
tests without internet access (one per architecture used in the experiments):

  bert       BERT + WordPiece                  (BioBERT path)
  modernbert ModernBERT + byte-level BPE        (Bio-ModernBERT path)
  t5         T5 + SentencePiece unigram         (SciFive path)
  t5gemma    T5Gemma + SentencePiece BPE (Gemma ids: pad 0, eos 1, bos 2, unk 3)
  llm        tiny Qwen2 causal LM + chat template (LLM HF backend)

    python scripts/smoke/make_tiny_models.py --output .smoke/models
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phee.data import load_samples  # noqa: E402

CHAT_TEMPLATE = (
    "{% for message in messages %}<|{{ message['role'] }}|>\n{{ message['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def corpus(data_dir):
    texts = []
    for split in ("train", "dev", "test"):
        for sample in load_samples(Path(data_dir) / f"{split}.jsonl"):
            texts.append(sample["context"])
    texts += [
        "question: What are the events? context:",
        "[ADE] [PTE] [Adverse event] [Potential therapeutic event] [Subject] [Treatment] [Effect] none ;",
        "Trigger in adverse event. Treatment.Drug in potential therapeutic event.",
    ]
    return texts


def make_bert(texts, out):
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertModel, BertTokenizerFast

    trainer = BertWordPieceTokenizer(lowercase=False)
    trainer.train_from_iterator(texts, vocab_size=2000, min_frequency=1)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer = BertTokenizerFast(vocab_file=str(out / "vocab.txt"), do_lower_case=False)
    tokenizer.save_pretrained(out)
    config = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=512)
    BertModel(config).save_pretrained(out)


def make_modernbert(texts, out):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

    specials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=2000, special_tokens=specials))
    cls_id, sep_id = tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
        pad_token="[PAD]", mask_token="[MASK]",
    )
    out.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(out)
    config = ModernBertConfig(
        vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=512, pad_token_id=fast.pad_token_id,
        cls_token_id=fast.cls_token_id, sep_token_id=fast.sep_token_id,
        bos_token_id=fast.cls_token_id, eos_token_id=fast.sep_token_id,
        global_attn_every_n_layers=1, local_attention=64,
    )
    ModernBertModel(config).save_pretrained(out)


def _sentencepiece(texts, out, model_type, gemma_ids):
    import sentencepiece as spm

    out.mkdir(parents=True, exist_ok=True)
    corpus_file = out / "corpus.txt"
    corpus_file.write_text("\n".join(texts), encoding="utf-8")
    kwargs = dict(input=str(corpus_file), model_prefix=str(out / "spiece"), vocab_size=1500,
                  model_type=model_type, character_coverage=1.0)
    if gemma_ids:
        kwargs.update(pad_id=0, eos_id=1, bos_id=2, unk_id=3, byte_fallback=True,
                      add_dummy_prefix=False, user_defined_symbols=["<mask>"])
    else:
        kwargs.update(pad_id=0, eos_id=1, unk_id=2, bos_id=-1)
    spm.SentencePieceTrainer.train(**kwargs)
    corpus_file.unlink()
    return out / "spiece.model"


def make_t5(texts, out):
    from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

    model_file = _sentencepiece(texts, out, "unigram", gemma_ids=False)
    tokenizer = T5Tokenizer(vocab_file=str(model_file), extra_ids=0, legacy=False)
    tokenizer.save_pretrained(out)
    config = T5Config(vocab_size=len(tokenizer), d_model=32, d_kv=16, d_ff=64, num_layers=2,
                      num_decoder_layers=2, num_heads=2, decoder_start_token_id=tokenizer.pad_token_id,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    T5ForConditionalGeneration(config).save_pretrained(out)


def make_t5gemma(texts, out):
    from transformers import GemmaTokenizer, T5GemmaConfig, T5GemmaForConditionalGeneration, T5GemmaModuleConfig

    model_file = _sentencepiece(texts, out, "bpe", gemma_ids=True)
    tokenizer = GemmaTokenizer(vocab_file=str(model_file), pad_token="<pad>", eos_token="</s>",
                               bos_token="<s>", unk_token="<unk>", add_bos_token=True, add_eos_token=False)
    tokenizer.save_pretrained(out)
    module = dict(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=512,
                  sliding_window=64, pad_token_id=0, eos_token_id=1, bos_token_id=2)
    config = T5GemmaConfig(encoder=T5GemmaModuleConfig(**module), decoder=T5GemmaModuleConfig(**module),
                           vocab_size=len(tokenizer), pad_token_id=0, eos_token_id=1, bos_token_id=2,
                           decoder_start_token_id=2)
    model = T5GemmaForConditionalGeneration(config)
    model.generation_config.decoder_start_token_id = 2
    model.generation_config.bos_token_id = 2
    model.generation_config.eos_token_id = 1
    model.generation_config.pad_token_id = 0
    model.save_pretrained(out)


def make_llm(texts, out):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    specials = ["<unk>", "<s>", "</s>", "<pad>"]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts + ['{"events": [{"event_type": "ADE", "trigger": ""}]}'],
                                  trainers.BpeTrainer(vocab_size=2000, special_tokens=specials))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>",
                                   eos_token="</s>", pad_token="<pad>")
    fast.chat_template = CHAT_TEMPLATE
    out.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(out)
    config = Qwen2Config(vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=8192,
                         bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id,
                         pad_token_id=fast.pad_token_id)
    Qwen2ForCausalLM(config).save_pretrained(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/phee2/processed")
    parser.add_argument("--output", default=".smoke/models")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    texts = corpus(args.data_dir)
    out = Path(args.output)
    builders = {"bert": make_bert, "modernbert": make_modernbert, "t5": make_t5,
                "t5gemma": make_t5gemma, "llm": make_llm}
    for name, builder in builders.items():
        if args.only and name not in args.only:
            continue
        builder(texts, out / name)
        print(f"built {out / name}")
    (out / "README.json").write_text(json.dumps({"note": "random weights, for smoke tests only"}))


if __name__ == "__main__":
    main()

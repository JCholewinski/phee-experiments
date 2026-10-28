"""Loading of Hugging Face models and tokenizers (pretrained ids or local directories)."""


def load_tokenizer(name, **kwargs):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, **kwargs)


def load_encoder(name, attn_implementation=None, **config_overrides):
    """Bare encoder (AutoModel) for token tagging / classification."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(name, **config_overrides)
    kwargs = {"config": config}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModel.from_pretrained(name, **kwargs)


def model_kwargs(model_config):
    """Loading arguments shared by all model types (passed to `from_pretrained` or
    `load_encoder`).

    ModernBERT runs its embeddings and MLPs through `torch.compile` on GPU by default
    (`reference_compile`). Compilation needs a C compiler in the image, takes time in
    every new process and only pays off for long inputs, so it is switched off unless
    the config says `model.compile: true`.
    """
    from transformers import AutoConfig

    kwargs = {}
    if model_config.get("attn_implementation"):
        kwargs["attn_implementation"] = model_config["attn_implementation"]
    config = AutoConfig.from_pretrained(model_config["name"])
    if hasattr(config, "reference_compile"):
        kwargs["reference_compile"] = bool(model_config.get("compile", False))
    return kwargs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

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
    """Optional loading arguments shared by all model types."""
    kwargs = {}
    if model_config.get("attn_implementation"):
        kwargs["attn_implementation"] = model_config["attn_implementation"]
    return kwargs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

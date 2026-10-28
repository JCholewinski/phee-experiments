"""PHEE information extraction experiments.

Package layout:
    schema      label inventory shared by all methods
    data        conversion of the revised PHEE data (Sun et al. 2024) to one unified format
    evaluation  text-based evaluation (unique (label, normalized text) pairs per sentence)
    training    common training loop with early stopping, seeds, timing and GPU memory
    seq         sequence labeling (authors' baseline and own two-model variants, heads, decoding)
    eqa         extractive QA pipeline (event type -> trigger -> arguments -> sub-arguments)
    gqa         generative QA pipeline (stage 1 events, stage 2 sub-arguments, constrained decoding)
    llm         zero-shot / few-shot extraction with local open-weight LLMs
    analysis    aggregation of results over seeds and variants
"""

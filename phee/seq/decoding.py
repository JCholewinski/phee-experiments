"""Decoding of word-level emissions into tag sequences."""

import torch

from phee.seq.model import NEG_INF, bio_constraints


def argmax_decode(emissions, word_mask):
    predictions = emissions.argmax(-1).cpu().tolist()
    lengths = word_mask.long().sum(1).cpu().tolist()
    return [row[:length] for row, length in zip(predictions, lengths)]


def viterbi(emissions, word_mask, transitions, start, end):
    """Batched Viterbi. emissions (B, T, C); returns list of tag-id lists."""
    emissions = emissions.float()
    batch, length, num_tags = emissions.shape
    score = start.unsqueeze(0) + emissions[:, 0]
    history = []
    for t in range(1, length):
        candidates = score.unsqueeze(2) + transitions.unsqueeze(0)
        best_score, best_previous = candidates.max(dim=1)
        next_score = best_score + emissions[:, t]
        active = word_mask[:, t].unsqueeze(1)
        score = torch.where(active, next_score, score)
        history.append(torch.where(active, best_previous, torch.arange(num_tags, device=score.device).expand(batch, -1)))
    score = score + end.unsqueeze(0)

    lengths = word_mask.long().sum(1).tolist()
    best_last = score.argmax(1).tolist()
    paths = []
    history = [h.cpu() for h in history]
    for b in range(batch):
        if lengths[b] == 0:
            paths.append([])
            continue
        tag = best_last[b]
        path = [tag]
        for t in range(lengths[b] - 1, 0, -1):
            tag = int(history[t - 1][b, tag])
            path.append(tag)
        path.reverse()
        paths.append(path)
    return paths


def decode(model, emissions, word_mask, strategy):
    """strategy: "argmax" or "viterbi" (BIO-constrained; learned transitions for CRF)."""
    if strategy == "argmax":
        return argmax_decode(emissions, word_mask)
    if strategy != "viterbi":
        raise ValueError(f"Unknown decoding: {strategy}")

    device = emissions.device
    if model.head_kind == "crf":
        transitions, start, end = model.head.crf.scores()
        allowed, allowed_start = bio_constraints(model.tag_names)
        transitions = transitions.masked_fill(~allowed.to(device), NEG_INF)
        start = start.masked_fill(~allowed_start.to(device), NEG_INF)
        scores = emissions.float()
    else:
        allowed, allowed_start = bio_constraints(model.tag_names)
        transitions = torch.zeros(allowed.shape, device=device).masked_fill(~allowed.to(device), NEG_INF)
        start = torch.zeros(allowed_start.shape, device=device).masked_fill(~allowed_start.to(device), NEG_INF)
        end = torch.zeros(allowed_start.shape, device=device)
        scores = torch.log_softmax(emissions.float(), dim=-1)
    return viterbi(scores, word_mask, transitions.float(), start.float(), end.float())


def native_decoding(model):
    """Decoding used for checkpoint selection: Viterbi for CRF, argmax otherwise."""
    return "viterbi" if model.head_kind == "crf" else "argmax"

"""Token tagger: any HF encoder (BioBERT, Bio-ModernBERT, ...) + a prediction head.

The model works on WORDS (the PHEE tokens): every word is represented by its first
sub-word. Heads:
  linear  dropout -> Linear(hidden, tags)                         (standard)
  mlp     dropout -> Linear(hidden, m) -> act -> dropout -> Linear(m, tags)
  crf     linear emissions + a linear-chain CRF with LEARNED transition scores,
          trained with the CRF negative log-likelihood. With `crf_constrained`,
          transitions that are invalid in the BIO scheme (O -> I-X, B-X -> I-Y, start
          -> I-X) are forbidden during training and decoding.

Decoding (phee.seq.decoding) is independent of the head:
  argmax   best tag per word
  viterbi  best tag SEQUENCE under BIO constraints; for linear/MLP the transition
           scores are 0 (only the constraints act), for CRF the learned ones are used.
"""

from dataclasses import dataclass

import torch
from torch import nn

NEG_INF = -1e4


def bio_constraints(tag_names):
    """allowed[i, j] (transition i -> j), allowed_start[j]."""
    size = len(tag_names)
    allowed = torch.ones(size, size, dtype=torch.bool)
    allowed_start = torch.ones(size, dtype=torch.bool)
    if not any(tag.startswith("B-") for tag in tag_names):
        return allowed, allowed_start  # IO scheme: every transition is valid
    for j, tag in enumerate(tag_names):
        if tag.startswith("I-"):
            entity = tag[2:]
            allowed_start[j] = False
            for i, previous in enumerate(tag_names):
                if previous not in (f"B-{entity}", f"I-{entity}"):
                    allowed[i, j] = False
    return allowed, allowed_start


class CRF(nn.Module):
    """Linear-chain CRF over contiguous (left-aligned) sequences."""

    def __init__(self, num_tags, allowed=None, allowed_start=None):
        super().__init__()
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        if allowed is None:
            allowed = torch.ones(num_tags, num_tags, dtype=torch.bool)
            allowed_start = torch.ones(num_tags, dtype=torch.bool)
        self.register_buffer("allowed", allowed.clone())
        self.register_buffer("allowed_start", allowed_start.clone())

    def scores(self):
        transitions = self.transitions.masked_fill(~self.allowed, NEG_INF)
        start = self.start_transitions.masked_fill(~self.allowed_start, NEG_INF)
        return transitions, start, self.end_transitions

    def neg_log_likelihood(self, emissions, tags, mask):
        """emissions (B, T, C) float, tags (B, T) long (any value where mask is 0),
        mask (B, T) bool, every row starts with True and is left-aligned."""
        emissions = emissions.float()
        transitions, start, end = self.scores()
        batch, length, _ = emissions.shape
        tags = tags.clamp(min=0)
        mask_f = mask.float()

        # score of the gold path
        score = start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, length):
            step = transitions[tags[:, t - 1], tags[:, t]] + emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + step * mask_f[:, t]
        lengths = mask.long().sum(1)
        last_tags = tags.gather(1, (lengths - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        score = score + end[last_tags]

        # log partition
        alpha = start.unsqueeze(0) + emissions[:, 0]
        for t in range(1, length):
            next_alpha = torch.logsumexp(
                alpha.unsqueeze(2) + transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1), dim=1
            )
            alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)
        log_z = torch.logsumexp(alpha + end.unsqueeze(0), dim=1)

        has_words = lengths > 0
        nll = (log_z - score)[has_words]
        return nll.sum() / max(1, int(lengths.sum()))


class Head(nn.Module):
    def __init__(self, hidden_size, num_tags, kind, dropout, mlp_hidden_size, mlp_dropout,
                 mlp_activation, tag_names, crf_constrained):
        super().__init__()
        self.kind = kind
        self.dropout = nn.Dropout(dropout)
        if kind == "mlp":
            activation = {"gelu": nn.GELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[mlp_activation]
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_size),
                activation,
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden_size, num_tags),
            )
        elif kind in ("linear", "crf"):
            self.classifier = nn.Linear(hidden_size, num_tags)
        else:
            raise ValueError(f"Unknown head: {kind}")
        if kind == "crf":
            allowed, allowed_start = (bio_constraints(tag_names) if crf_constrained else (None, None))
            self.crf = CRF(num_tags, allowed, allowed_start)

    def forward(self, word_hidden):
        return self.classifier(self.dropout(word_hidden))


@dataclass
class TaggerOutput:
    loss: torch.Tensor = None
    emissions: torch.Tensor = None
    word_mask: torch.Tensor = None


class TokenTagger(nn.Module):
    def __init__(self, encoder, tag_names, head="linear", dropout=0.1, mlp_hidden_size=256,
                 mlp_dropout=0.1, mlp_activation="gelu", crf_constrained=True):
        super().__init__()
        self.encoder = encoder
        self.tag_names = list(tag_names)
        self.head_kind = head
        self.head = Head(
            hidden_size=encoder.config.hidden_size,
            num_tags=len(tag_names),
            kind=head,
            dropout=dropout,
            mlp_hidden_size=mlp_hidden_size,
            mlp_dropout=mlp_dropout,
            mlp_activation=mlp_activation,
            tag_names=tag_names,
            crf_constrained=crf_constrained,
        )

    def forward(self, input_ids, attention_mask, word_positions, word_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        index = word_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        word_hidden = hidden.gather(1, index)
        emissions = self.head(word_hidden)

        loss = None
        if labels is not None:
            if self.head_kind == "crf":
                loss = self.head.crf.neg_log_likelihood(emissions, labels, word_mask)
            else:
                loss = nn.functional.cross_entropy(
                    emissions.float().reshape(-1, emissions.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
        return TaggerOutput(loss=loss, emissions=emissions, word_mask=word_mask)

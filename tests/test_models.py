"""Tests of the CRF and the decoding against brute-force enumeration (needs torch).

    python -m pytest tests -q        or        python tests/test_models.py
"""

import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phee.seq.decoding import viterbi  # noqa: E402
from phee.seq.model import CRF, NEG_INF, bio_constraints  # noqa: E402

TAGS = ["O", "B-X", "I-X", "B-Y", "I-Y"]


def _path_score(emissions, tags, transitions, start, end):
    score = start[tags[0]] + emissions[0, tags[0]]
    for t in range(1, len(tags)):
        score = score + transitions[tags[t - 1], tags[t]] + emissions[t, tags[t]]
    return score + end[tags[-1]]


def test_bio_constraints():
    allowed, allowed_start = bio_constraints(TAGS)
    index = {tag: i for i, tag in enumerate(TAGS)}
    assert not allowed[index["O"], index["I-X"]]
    assert allowed[index["B-X"], index["I-X"]]
    assert allowed[index["I-X"], index["I-X"]]
    assert not allowed[index["B-Y"], index["I-X"]]
    assert allowed[index["I-X"], index["B-Y"]]
    assert not allowed_start[index["I-Y"]] and allowed_start[index["B-Y"]]
    io_allowed, io_start = bio_constraints(["O", "I-X", "I-Y"])
    assert io_allowed.all() and io_start.all()


def test_crf_nll_matches_enumeration():
    torch.manual_seed(0)
    num_tags = 3
    crf = CRF(num_tags)
    with torch.no_grad():
        crf.transitions.normal_()
        crf.start_transitions.normal_()
        crf.end_transitions.normal_()
    lengths = [4, 2]
    emissions = torch.randn(2, 4, num_tags)
    tags = torch.tensor([[0, 2, 1, 1], [2, 0, -100, -100]])
    mask = torch.tensor([[True] * 4, [True, True, False, False]])
    nll = crf.neg_log_likelihood(emissions, tags, mask)

    transitions, start, end = crf.scores()
    expected = 0.0
    for row, length in enumerate(lengths):
        e = emissions[row, :length]
        all_scores = torch.stack([
            _path_score(e, list(path), transitions, start, end)
            for path in itertools.product(range(num_tags), repeat=length)
        ])
        log_z = torch.logsumexp(all_scores, 0)
        gold = _path_score(e, tags[row, :length].tolist(), transitions, start, end)
        expected += float(log_z - gold)
    expected /= sum(lengths)
    assert abs(float(nll) - expected) < 1e-4, (float(nll), expected)


def test_viterbi_matches_enumeration():
    torch.manual_seed(1)
    num_tags = len(TAGS)
    allowed, allowed_start = bio_constraints(TAGS)
    transitions = torch.randn(num_tags, num_tags).masked_fill(~allowed, NEG_INF)
    start = torch.randn(num_tags).masked_fill(~allowed_start, NEG_INF)
    end = torch.randn(num_tags)
    emissions = torch.randn(3, 5, num_tags)
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2, [True] + [False] * 4])
    paths = viterbi(emissions, mask, transitions, start, end)
    for row, length in enumerate([5, 3, 1]):
        e = emissions[row, :length]
        best = max(
            itertools.product(range(num_tags), repeat=length),
            key=lambda p: float(_path_score(e, list(p), transitions, start, end)),
        )
        assert paths[row] == list(best), (row, paths[row], best)


def test_constrained_viterbi_repairs_invalid_bio():
    from phee.seq.decoding import decode

    class Dummy:
        head_kind = "linear"
        tag_names = TAGS

    # argmax would give O I-X I-X (invalid start of an entity)
    emissions = torch.log(torch.tensor([[[0.6, 0.0, 0.4, 0.0, 0.0],
                                         [0.0, 0.3, 0.7, 0.0, 0.0],
                                         [0.0, 0.0, 1.0, 0.0, 0.0]]]) + 1e-6)
    mask = torch.ones(1, 3, dtype=torch.bool)
    argmax = decode(Dummy(), emissions, mask, "argmax")[0]
    constrained = decode(Dummy(), emissions, mask, "viterbi")[0]
    assert [TAGS[i] for i in argmax] == ["O", "I-X", "I-X"]
    # best valid sequence: O (0.6) B-X (0.3) I-X (1.0)
    assert [TAGS[i] for i in constrained] == ["O", "B-X", "I-X"]


def test_trigger_margin_selection():
    from phee.eqa.runner import _within_margin

    triggers = [[
        {"event_type": "ADE", "text": "a", "score": 10.0},
        {"event_type": "ADE", "text": "b", "score": 9.2},
        {"event_type": "ADE", "text": "c", "score": 6.0},
        {"event_type": "PTE", "text": "d", "score": 3.0},
    ]]
    assert [t["text"] for t in _within_margin(triggers, None)[0]] == ["a", "d"]
    assert [t["text"] for t in _within_margin(triggers, 1.0)[0]] == ["a", "b", "d"]
    assert [t["text"] for t in _within_margin(triggers, 5.0)[0]] == ["a", "b", "c", "d"]


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"ok   {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL {name}: {error}")
    sys.exit(1 if failures else 0)

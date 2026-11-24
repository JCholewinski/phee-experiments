"""The authors' EEQA answer rule (eeqa/code/run_stage1.py, make_predictions)."""

from phee.eqa.runner import authors_overlap_filter


def _a(start, end, text="x"):
    return {"start": start, "end": end, "text": text, "score": 0.0}


def test_overlap_filter_follows_the_authors_code():
    ade_a = {"event_type": "ADE", "text": "developed"}
    ade_b = {"event_type": "ADE", "text": "after"}
    pte = {"event_type": "PTE", "text": "improved"}
    results = [
        # 2 best candidates of one question: the second overlaps the first -> one answer
        (0, ("TREATMENT", ade_a), [_a(0, 9, "isoniazid"), _a(0, 17, "isoniazid therapy")]),
        # another trigger of the same type: the same span again -> dropped
        (0, ("TREATMENT", ade_b), [_a(0, 9, "isoniazid")]),
        # other event type -> its own spans
        (0, ("TREATMENT", pte), [_a(0, 9, "isoniazid")]),
        # other role -> its own spans
        (0, ("EFFECT", ade_a), [_a(30, 39, "hepatitis"), _a(50, 60, "jaundice")]),
        # other sentence -> its own spans
        (1, ("TREATMENT", ade_a), [_a(0, 9, "isoniazid")]),
        # their check looks only at the start and end positions: a span containing a kept
        # one is not an overlap
        (0, ("TREATMENT", ade_b), [_a(0, 25, "isoniazid therapy for TB")]),
    ]
    out = authors_overlap_filter(results, key=lambda meta: (meta[1]["event_type"], meta[0]))
    assert [[a["text"] for a in answers] for _, _, answers in out] == [
        ["isoniazid"], [], ["isoniazid"], ["hepatitis", "jaundice"], ["isoniazid"], [],
    ]


def test_containing_span_is_kept_like_in_their_code():
    meta = ("TREATMENT", {"event_type": "ADE"})
    results = [(0, meta, [_a(5, 10)]), (0, meta, [_a(0, 20)])]
    out = authors_overlap_filter(results, key=lambda m: (m[1]["event_type"], m[0]))
    assert [len(answers) for _, _, answers in out] == [1, 1]

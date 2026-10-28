"""Text normalization identical to the authors' official evaluator
(PHEE repository, evaluate/phee_metric.py, function _normalize_answer).

lower case -> punctuation turned into token boundaries -> articles removed -> whitespace fixed
  "A 52-year-old woman"  -> "52 year old woman"
  "50% solution"         -> "50 solution"
"""

import re
import string

_PUNCTUATION = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _remove_punctuation(text):
    pieces = []
    for word in text.split():
        current = ""
        for char in word:
            if char not in _PUNCTUATION:
                current += char
            elif current:
                pieces.append(current)
                current = ""
        if current:
            pieces.append(current)
    return " ".join(pieces)


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = _remove_punctuation(text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())

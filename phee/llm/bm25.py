"""Okapi BM25 retrieval of few-shot demonstrations (Sun et al. 2024: best selection)."""

import math
from collections import Counter

from phee.normalize import normalize_text


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(normalize_text(doc).split()) for doc in documents]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency = Counter()
        for doc in self.docs:
            document_frequency.update(doc.keys())
        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def scores(self, query):
        terms = normalize_text(query).split()
        result = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for term in terms:
                frequency = doc.get(term)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                score += self.idf[term] * frequency * (self.k1 + 1) / denominator
            result.append(score)
        return result

    def top(self, query, k):
        scores = self.scores(query)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

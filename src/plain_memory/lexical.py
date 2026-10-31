"""Okapi-style BM25 over per-user term lists, in plain Python.

Scores follow the common "Lucene" variant: idf = ln(1 + (N - df + 0.5) / (df + 0.5)) and
tf / (tf + k1 * (1 - b + b * len / avg_len)). The idf is always positive, so a document that
contains a query term always scores above zero, even in a one-document corpus.
"""

import heapq
import math
from collections import Counter, defaultdict


class BM25Index:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.size = len(documents)
        lengths = [len(terms) for terms in documents]
        average = sum(lengths) / self.size if self.size else 0.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc, terms in enumerate(documents):
            for term, count in Counter(terms).items():
                postings[term].append((doc, count))
        self.postings = dict(postings)
        # Document length enters the score only through this per-document constant.
        self.damping = [k1 * (1 - b + b * n / average) if average else k1 for n in lengths]
        self.idf = {
            term: math.log(1 + (self.size - len(hits) + 0.5) / (len(hits) + 0.5))
            for term, hits in self.postings.items()
        }

    def scores(self, query_terms: list[str]) -> dict[int, float]:
        """Score of every document sharing at least one term with the query."""
        total: dict[int, float] = defaultdict(float)
        for term in query_terms:
            hits = self.postings.get(term)
            if not hits:
                continue
            weight = self.idf[term]
            for doc, count in hits:
                total[doc] += weight * count / (count + self.damping[doc])
        return total

    def top(self, query_terms: list[str], limit: int) -> list[int]:
        """Best `limit` documents; equal scores keep document order."""
        found = self.scores(query_terms)
        return heapq.nsmallest(max(0, limit), found, key=lambda doc: (-found[doc], doc))

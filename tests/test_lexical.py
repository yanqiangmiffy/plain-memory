import math

import pytest

from plain_memory.lexical import BM25Index


def lucene(df, size, tf, length, average, k1=1.5, b=0.75):
    idf = math.log(1 + (size - df + 0.5) / (df + 0.5))
    return idf * tf / (tf + k1 * (1 - b + b * length / average))


def test_scores_match_the_closed_form():
    docs = [["red", "red", "fox"], ["red", "dog"], ["blue", "dog", "dog", "cat"]]
    index = BM25Index(docs)
    average = sum(len(d) for d in docs) / 3
    got = index.scores(["red", "dog"])
    expected_0 = lucene(2, 3, 2, 3, average)
    expected_1 = lucene(2, 3, 1, 2, average) + lucene(2, 3, 1, 2, average)
    expected_2 = lucene(2, 3, 2, 4, average)
    assert got[0] == pytest.approx(expected_0)
    assert got[1] == pytest.approx(expected_1)
    assert got[2] == pytest.approx(expected_2)


def test_only_documents_sharing_a_term_are_returned():
    index = BM25Index([["a", "b"], ["c"], ["b", "d"]])
    assert set(index.scores(["b"])) == {0, 2}
    assert index.scores(["zzz"]) == {}
    assert index.top(["zzz"], 5) == []


def test_single_document_corpus_still_scores_positive():
    index = BM25Index([["only", "doc"]])
    assert index.top(["only"], 3) == [0]
    assert index.scores(["only"])[0] > 0


def test_equal_scores_keep_document_order_and_limit_is_respected():
    index = BM25Index([["x"], ["x"], ["x"], ["y"]])
    assert index.top(["x"], 2) == [0, 1]
    assert index.top(["x"], 10) == [0, 1, 2]
    assert index.top(["x"], 0) == []


def test_repeated_query_terms_count_each_time():
    index = BM25Index([["a", "b"], ["b", "b"]])
    assert index.scores(["b", "b"])[0] == pytest.approx(2 * index.scores(["b"])[0])


def test_empty_corpus_is_harmless():
    index = BM25Index([])
    assert index.top(["a"], 5) == []

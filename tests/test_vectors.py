import math
import random
import struct
from concurrent.futures import ThreadPoolExecutor

import pytest

from plain_memory.embedding import EmbeddingError, normalize_vectors
from plain_memory.vectors import DenseIndex, pack_vector, unpack_vector


def test_faiss_matches_scalar_inner_product_and_parallel_queries():
    rng = random.Random(17)
    vectors = normalize_vectors([[rng.uniform(-1, 1) for _ in range(32)] for _ in range(200)], 200)
    index = DenseIndex(pack_vector(v) for v in vectors)
    queries = vectors[:12]
    expected = [
        sorted(
            range(len(vectors)),
            key=lambda i: (-math.fsum(a * b for a, b in zip(vectors[i], q)), i),
        )[:17]
        for q in queries
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda q: index.search(q, 17), queries))
    assert actual == expected


def test_faiss_keeps_insertion_order_for_ties_across_top_k_boundary():
    index = DenseIndex(pack_vector(v) for v in [[1, 0]] * 200 + [[0, 1]])
    assert index.search([1, 0], 7) == list(range(7))
    assert index.search([1, 0], 1000) == list(range(201))
    assert index.search([1, 0], 0) == []


@pytest.mark.parametrize(
    "blobs",
    [
        [],
        [b""],
        [b"bad"],
        [pack_vector([0, 0])],
        [pack_vector([math.nan, 1])],
        [pack_vector([math.inf, 1])],
        [pack_vector([1, 0]), pack_vector([1])],
    ],
)
def test_faiss_rejects_invalid_storage_before_native_calls(blobs):
    with pytest.raises(ValueError):
        DenseIndex(blobs)


@pytest.mark.parametrize("query", [[], [1], [1, 2, 3], [0, 0], [math.nan, 1], [math.inf, 1]])
def test_faiss_rejects_invalid_queries(query):
    index = DenseIndex([pack_vector([1, 0])])
    with pytest.raises(ValueError):
        index.search(query, 5)


def test_float_buffers_read_and_write_legacy_little_endian_blobs():
    blob = struct.pack("<4f", 0.5, -0.25, 0.125, 0.0)
    assert list(unpack_vector(blob)) == [0.5, -0.25, 0.125, 0.0]
    assert pack_vector(unpack_vector(blob)) == blob
    assert DenseIndex([blob]).search([1, 0, 0, 0], 100) == [0]


@pytest.mark.parametrize(
    "values,count,dimensions",
    [
        ([], 0, None),
        ([[1, 2]], 2, None),
        ([[1, 2]], 1, 3),
        ([[1], [1, 2]], 2, None),
        ([[]], 1, None),
        ([[0, 0]], 1, None),
        ([[float("nan"), 1]], 1, None),
        ([[float("inf"), 1]], 1, None),
        ([[1e100, 1]], 1, None),
        ([1, 2], 2, None),
    ],
)
def test_invalid_embedding_shapes_and_values_fail(values, count, dimensions):
    with pytest.raises(EmbeddingError):
        normalize_vectors(values, count, dimensions)


def test_normalization_handles_small_finite_float32_values():
    (vector,) = normalize_vectors([[1e-30, 1e-30]], 1, 2)
    assert math.hypot(*vector) == pytest.approx(1.0)

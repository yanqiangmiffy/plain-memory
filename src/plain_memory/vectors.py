"""Float32 buffers and exact FAISS CPU search without application-level NumPy.

SQLite keeps the existing little-endian float32 format. FAISS owns its index;
temporary native-endian stdlib arrays stay alive throughout each SWIG call.
"""

import math
import sys
from array import array
from itertools import chain

import faiss


def unpack_vector(blob: bytes) -> array:
    vector = array("f")
    vector.frombytes(blob)
    if sys.byteorder != "little":
        vector.byteswap()
    if not _valid_vector(vector):
        raise ValueError("Invalid stored vector")
    return vector


def pack_vector(vector) -> bytes:
    values = array("f", vector)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _float_pointer(buffer: array):
    if buffer.typecode != "f" or buffer.itemsize != 4 or not buffer:
        raise ValueError("Expected a nonempty native float32 buffer")
    return faiss.cast_integer_to_float_ptr(buffer.buffer_info()[0])


def _valid_vector(vector: array) -> bool:
    if not vector:
        return False
    # A finite, positive squared norm proves the buffer is finite and nonzero.
    # Let FAISS scan normal embeddings in native code. Keep a scalar fallback
    # for finite vectors whose float32 squared norm underflows or overflows.
    norm = faiss.fvec_norm_L2sqr(_float_pointer(vector), len(vector))
    return (math.isfinite(norm) and norm > 0) or (
        all(math.isfinite(x) for x in vector) and any(vector)
    )


class DenseIndex:
    """Exact inner-product search over already normalized embeddings.

    Keep insertion order for equal scores, including ties at the top-k boundary.
    One FAISS thread per request avoids multiplying the service's concurrency
    by the machine's default OpenMP thread count.
    """

    def __init__(self, blobs):
        self.index = None
        batch = array("f")
        for blob in blobs:
            vector = unpack_vector(blob)
            if self.index is None:
                self.index = faiss.IndexFlatIP(len(vector))
            if len(vector) != self.index.d:
                raise ValueError("Stored vectors have inconsistent dimensions")
            batch.extend(vector)
            if len(batch) // self.index.d >= 1024:
                self.index.add_c(len(batch) // self.index.d, _float_pointer(batch))
                batch = array("f")
        if self.index is None:
            raise ValueError("Cannot index an empty corpus")
        if batch:
            self.index.add_c(len(batch) // self.index.d, _float_pointer(batch))

    def search(self, vector, limit: int) -> list[int]:
        query = array("f", vector)
        if len(query) != self.index.d:
            raise ValueError("Query dimension differs from stored vectors")
        if not _valid_vector(query):
            raise ValueError("Invalid query vector")
        limit = min(limit, self.index.ntotal)
        if limit <= 0:
            return []
        count = min(limit + 1, self.index.ntotal)
        previous_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(1)
        try:
            while True:
                distances = array("f", [0]) * count
                labels = array("q", [0]) * count
                if labels.itemsize != 8:
                    raise ValueError("FAISS requires 64-bit labels")
                self.index.search_c(
                    1,
                    _float_pointer(query),
                    count,
                    _float_pointer(distances),
                    faiss.cast_integer_to_idx_t_ptr(labels.buffer_info()[0]),
                )
                # FAISS can choose any tied label. Expand until the complete
                # boundary tie is available before imposing a stable order.
                if count == self.index.ntotal or distances[limit - 1] != distances[-1]:
                    return [
                        label
                        for _, label in sorted(zip(distances, labels), key=lambda p: (-p[0], p[1]))
                    ][:limit]
                count = min(count * 2, self.index.ntotal)
        finally:
            faiss.omp_set_num_threads(previous_threads)


def normalize(values, count: int, dimensions: int | None = None) -> list[array]:
    rows = list(values)
    if count <= 0 or len(rows) != count:
        raise ValueError("Invalid vector count")
    vectors = [array("f", row) for row in rows]
    width = dimensions if dimensions is not None else len(vectors[0])
    if width <= 0 or any(len(row) != width for row in vectors):
        raise ValueError("Invalid vector dimensions")
    if not all(math.isfinite(x) for x in chain.from_iterable(vectors)):
        raise ValueError("Non-finite vector")
    result = []
    for row in vectors:
        norm = math.hypot(*row)
        if not norm:
            raise ValueError("Zero vector")
        result.append(array("f", (x / norm for x in row)))
    return result

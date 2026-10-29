"""Optional on-disk embedding cache (SQLite) keyed by model, dimension and exact text.

Off by default. It is meant for offline experiments, where the same history is ingested many
times under different retrieval settings. The file holds vectors derived from the texts, so in
a real evaluation it falls under the same 30-day deletion duty as the main database.
"""

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, path: Path, namespace: str):
        self.path = path
        self.namespace = namespace
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, vec BLOB NOT NULL)"
            )
            conn.commit()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA synchronous=OFF")  # disposable data
        return conn

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.namespace}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        with closing(self._connect()) as conn:
            for start in range(0, len(keys), 500):
                chunk = keys[start : start + 500]
                marks = ",".join("?" * len(chunk))
                for key, blob in conn.execute(
                    f"SELECT key, vec FROM cache WHERE key IN ({marks})", chunk
                ):
                    found[key] = np.frombuffer(blob, dtype="<f4")
        return found

    def put_many(self, items: dict[str, np.ndarray]):
        with closing(self._connect()) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO cache VALUES (?, ?)",
                [(key, np.asarray(vec, dtype="<f4").tobytes()) for key, vec in items.items()],
            )
            conn.commit()


class ScoreCache:
    """Optional on-disk cache of reranker scores keyed by (model, query, document)."""

    def __init__(self, path: Path, namespace: str):
        self.path = path
        self.namespace = namespace
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)"
            )
            conn.commit()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA synchronous=OFF")  # disposable data
        return conn

    def key(self, query: str, document: str) -> str:
        return hashlib.sha256(f"{self.namespace}\x00{query}\x00{document}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, float]:
        found: dict[str, float] = {}
        with closing(self._connect()) as conn:
            for start in range(0, len(keys), 500):
                chunk = keys[start : start + 500]
                marks = ",".join("?" * len(chunk))
                for key, score in conn.execute(
                    f"SELECT key, score FROM scores WHERE key IN ({marks})", chunk
                ):
                    found[key] = score
        return found

    def put_many(self, items: dict[str, float]):
        with closing(self._connect()) as conn:
            conn.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?)", list(items.items()))
            conn.commit()

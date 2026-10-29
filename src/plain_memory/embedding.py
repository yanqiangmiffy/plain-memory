"""只实现首版需要的两个 HTTP 编码格式，不记录请求或上游响应正文。"""

import time

import httpx
import numpy as np

from plain_memory.config import EmbeddingConfig
from plain_memory.embed_cache import EmbeddingCache

RETRY_ATTEMPTS = 5
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class EmbeddingError(Exception):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


def normalize_vectors(values, count: int, dimensions: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != count or array.shape[1] == 0:
            raise ValueError
        if dimensions is not None and array.shape[1] != dimensions:
            raise ValueError
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if not np.isfinite(array).all() or not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError
        return np.asarray(array / norms, dtype="<f4")
    except (ValueError, TypeError, OverflowError):
        raise EmbeddingError("Embedding returned invalid vectors or a changed dimension.") from None


class HTTPEmbedder:
    def __init__(self, config: EmbeddingConfig, *, transport=None):
        self.config = config
        key = config.api_key.get_secret_value()
        if not key or key.startswith("REPLACE_"):
            raise ValueError("Set embedding.api_key in the private local config.")
        if not config.base_url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
            raise ValueError("Embedding base_url must use HTTPS (or loopback HTTP for tests).")
        path = "/embeddings/multimodal" if config.protocol == "doubao_multimodal" else "/embeddings"
        self.url = config.base_url + path
        self.cache = None
        if config.cache_path is not None:
            namespace = f"{config.protocol}|{config.model}|{config.dimensions}"
            self.cache = EmbeddingCache(config.cache_path, namespace)
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            transport=transport,
            follow_redirects=False,
        )

    def close(self):
        self.client.close()

    def _post(self, payload: dict, deadline: float) -> httpx.Response:
        """对供应商限流/瞬时故障做有界退避重试，总时间不超过服务截止时间。"""
        delay = 1.0
        for attempt in range(RETRY_ATTEMPTS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EmbeddingError("Embedding request exceeded the service deadline.")
            try:
                response = self.client.post(
                    self.url,
                    json=payload,
                    timeout=httpx.Timeout(remaining, connect=min(5.0, remaining)),
                )
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == RETRY_ATTEMPTS - 1:
                    raise
            else:
                if response.status_code not in RETRYABLE_STATUS or attempt == RETRY_ATTEMPTS - 1:
                    return response
            if deadline - time.monotonic() <= delay:
                raise EmbeddingError("Embedding request exceeded the service deadline.")
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
        raise AssertionError("unreachable")

    def embed(self, texts: list[str], deadline: float) -> np.ndarray:
        if self.cache is None or not texts:
            return self._embed(texts, deadline)
        keys = [self.cache.key(text) for text in texts]
        found = self.cache.get_many(list(dict.fromkeys(keys)))
        pending = {key: text for key, text in zip(keys, texts) if key not in found}
        if pending:
            fresh = self._embed(list(pending.values()), deadline)
            added = dict(zip(pending.keys(), fresh))
            self.cache.put_many(added)
            found.update(added)
        return normalize_vectors([found[key] for key in keys], len(texts), self.config.dimensions)

    def _embed(self, texts: list[str], deadline: float) -> np.ndarray:
        vectors = []
        batch_size = 1 if self.config.protocol == "doubao_multimodal" else self.config.batch_size
        for start in range(0, len(texts), batch_size):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EmbeddingError("Embedding request exceeded the service deadline.")
            batch = texts[start : start + batch_size]
            if self.config.protocol == "doubao_multimodal":
                payload = {
                    "model": self.config.model,
                    "input": [{"type": "text", "text": batch[0]}],
                }
            else:
                payload = {"model": self.config.model, "input": batch}
                if self.config.dimensions is not None:
                    payload["dimensions"] = self.config.dimensions
            try:
                response = self._post(payload, deadline)
                if response.status_code in (400, 413, 422):
                    raise EmbeddingError(
                        "Embedding provider rejected input; check model limits.", 422
                    )
                response.raise_for_status()
                data = response.json()["data"]
                if self.config.protocol == "doubao_multimodal":
                    current = [data["embedding"]]
                else:
                    # 供应商返回顺序可能不同；按 index 对齐，并检查遗漏/重复 index。
                    indexes = [item["index"] for item in data]
                    if sorted(indexes) != list(range(len(batch))):
                        raise ValueError
                    current = [
                        item["embedding"] for item in sorted(data, key=lambda item: item["index"])
                    ]
                normalized = normalize_vectors(current, len(batch), self.config.dimensions)
                vectors.extend(normalized)
            except EmbeddingError:
                raise
            except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError):
                raise EmbeddingError(
                    "Embedding service unavailable or returned an invalid response."
                ) from None
        return normalize_vectors(vectors, len(texts), self.config.dimensions)

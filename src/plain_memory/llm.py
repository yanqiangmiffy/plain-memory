"""gpt-4o-mini JSON helper and DashScope reranker. Neither logs request/response bodies.

Both are best-effort enrichers: any failure returns None so Add/Search degrade to the
plain hybrid retrieval instead of failing the whole request.
"""

import json
import re
import threading
import time

import httpx

from plain_memory.config import LLMConfig, RerankConfig

RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


def _json_object(text: str):
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


class LLM:
    def __init__(self, config: LLMConfig, *, transport=None):
        self.config = config
        self.slots = threading.BoundedSemaphore(config.concurrency)
        self.client = httpx.Client(
            headers={"Authorization": "Bearer " + config.api_key.get_secret_value()},
            transport=transport,
            proxy=config.proxy,
            follow_redirects=False,
        )

    def close(self):
        self.client.close()

    def json(self, system: str, user: str, max_tokens: int, deadline: float):
        """Return the parsed JSON object produced by the model, or None."""
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        with self.slots:
            for attempt in range(4):
                remaining = min(deadline - time.monotonic(), self.config.timeout_seconds)
                if remaining <= 1:
                    return None
                try:
                    response = self.client.post(
                        self.config.base_url.rstrip("/") + "/chat/completions",
                        json=payload,
                        timeout=httpx.Timeout(remaining, connect=min(5.0, remaining)),
                    )
                    if response.status_code in RETRYABLE:
                        raise httpx.HTTPError("retryable")
                    if response.status_code != 200:
                        return None
                    return _json_object(response.json()["choices"][0]["message"]["content"])
                except (httpx.HTTPError, KeyError, IndexError, ValueError):
                    time.sleep(min(2 * (attempt + 1), max(0.0, deadline - time.monotonic())))
        return None


class Reranker:
    def __init__(self, config: RerankConfig, *, transport=None):
        self.config = config
        self.client = httpx.Client(
            headers={"Authorization": "Bearer " + config.api_key.get_secret_value()},
            transport=transport,
            follow_redirects=False,
        )

    def close(self):
        self.client.close()

    def scores(self, query: str, documents: list[str], deadline: float) -> list[float] | None:
        """Relevance score per document (same order), or None on any failure."""
        payload = {
            "model": self.config.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"return_documents": False, "top_n": len(documents)},
        }
        for attempt in range(3):
            remaining = min(deadline - time.monotonic(), self.config.timeout_seconds)
            if remaining <= 1:
                return None
            try:
                response = self.client.post(
                    self.config.base_url.rstrip("/") + "/services/rerank/text-rerank/text-rerank",
                    json=payload,
                    timeout=httpx.Timeout(remaining, connect=min(5.0, remaining)),
                )
                if response.status_code in RETRYABLE:
                    raise httpx.HTTPError("retryable")
                if response.status_code != 200:
                    return None
                result = [0.0] * len(documents)
                for item in response.json()["output"]["results"]:
                    result[item["index"]] = float(item["relevance_score"])
                return result
            except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError):
                time.sleep(min(1.5 * (attempt + 1), max(0.0, deadline - time.monotonic())))
        return None

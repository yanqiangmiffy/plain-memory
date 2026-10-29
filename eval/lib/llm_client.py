"""Sync chat client for the offline harness (answer + judge), with an on-disk response cache.

Defaults to OpenRouter gpt-4o-mini. For the free local vLLM server set
  EVAL_LLM_BASE=http://10.208.62.156:8004/v1  EVAL_LLM_MODEL=Qwen3.8-27B-FP8
(no key needed), or DeepSeek:
  EVAL_LLM_BASE=https://api.deepseek.com  EVAL_LLM_MODEL=deepseek-flash  (DEEPSEEK_API_KEY from .env)
Provider-specific fields go in EVAL_LLM_EXTRA (JSON). Results are cached by (base, model, prompt), so identical retrievals are free.
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import httpx

BASE = os.environ.get("EVAL_LLM_BASE", "https://openrouter.ai/api/v1").rstrip("/")
MODEL = os.environ.get("EVAL_LLM_MODEL", "openai/gpt-4o-mini")
EXTRA = json.loads(os.environ.get("EVAL_LLM_EXTRA", "{}"))  # provider-specific payload fields
KEY_VAR = (
    "OPENROUTER_API_KEY"
    if "openrouter" in BASE
    else "DEEPSEEK_API_KEY"
    if "deepseek" in BASE
    else None
)
_client = httpx.Client(timeout=600, trust_env="openrouter" in BASE)
_db_path = Path(__file__).resolve().parents[1] / ".cache" / "llm_cache.sqlite3"
_lock = threading.Lock()


def _cache_get(key: str):
    with _lock, sqlite3.connect(_db_path, timeout=30) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        row = conn.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
    return row[0] if row else None


def _cache_put(key: str, value: str):
    with _lock, sqlite3.connect(_db_path, timeout=30) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        conn.execute("INSERT OR REPLACE INTO c VALUES (?, ?)", (key, value))


def chat(prompt: str, max_tokens: int = 256, system: str | None = None) -> str:
    key = hashlib.sha256(f"{BASE}|{MODEL}|{max_tokens}|{system}|{prompt}".encode()).hexdigest()
    cached = _cache_get(key)
    if cached is not None:
        return cached
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        **EXTRA,
    }
    headers = {}
    if KEY_VAR:
        headers["Authorization"] = "Bearer " + os.environ[KEY_VAR]
    for attempt in range(8):
        try:
            r = _client.post(BASE + "/chat/completions", headers=headers, json=body)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPError(str(r.status_code))
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            _cache_put(key, text)
            return text
        except (httpx.HTTPError, KeyError):
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("LLM call failed")

"""Tiny sync gpt-4o-mini client for the offline harness (OpenRouter, OpenAI-compatible)."""

import os
import time

import httpx

BASE = os.environ.get("EVAL_LLM_BASE", "https://openrouter.ai/api/v1")
MODEL = os.environ.get("EVAL_LLM_MODEL", "openai/gpt-4o-mini")
_client = httpx.Client(timeout=180)


def chat(prompt: str, max_tokens: int = 256, system: str | None = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    for attempt in range(6):
        try:
            r = _client.post(
                BASE + "/chat/completions",
                headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
                json={"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens},
            )
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPError(str(r.status_code))
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError):
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("LLM call failed")

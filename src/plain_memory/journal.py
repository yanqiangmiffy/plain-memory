"""Optional append-only JSON-lines journal of every Add / Search call (request and response).

One JSON object per line:
  {"time", "endpoint", "status", "elapsed_ms", "request", "response", "error"}

It holds evaluation text, so it is created owner-only (0600), never leaves the machine, stops
growing at `max_bytes`, and falls under the same 30-day deletion duty as the database. The API
credential is never written: authentication happens before the handlers that record.
"""

import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from plain_memory.store import RequestConflict, StoreMismatch


def status_of(exc: BaseException) -> int:
    """HTTP status the app will answer with for an exception raised inside a handler."""
    code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(code, int):
        return code
    if isinstance(exc, RequestConflict):
        return 422
    if isinstance(exc, (StoreMismatch, sqlite3.Error)):
        return 503
    return 500


class Journal:
    def __init__(self, path: Path, max_bytes: int):
        self.path = path
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.full = False
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))

    def write(self, endpoint: str, started: float, status: int, request, response, error=None):
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "endpoint": endpoint,
            "status": status,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "request": request,
            "response": response,
            "error": error,
        }
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with self.lock:
            if self.full:
                return
            try:
                if self.path.stat().st_size + len(line) > self.max_bytes:
                    self.full = True
                    print(
                        "request journal reached its size limit; recording stopped", file=sys.stderr
                    )
                    return
                with self.path.open("ab") as handle:
                    handle.write(line)
            except OSError:
                pass  # the journal must never break the API

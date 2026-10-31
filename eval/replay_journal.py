"""Replay recorded searches against their historical SQLite prefix, without serving traffic.

Run twice with PYTHONPATH pointing to baseline/candidate src directories. The
database and journal are read-only. Query embedding cache/output stay local.
There is no answer model or inferred accuracy score in this diagnostic.
"""

import argparse
import hashlib
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from audit_journal import load_records, prior_adds

from plain_memory.api import SearchRequest
from plain_memory.config import load_settings
from plain_memory.embedding import HTTPEmbedder
from plain_memory.retrieval import UserIndex
from plain_memory.searching import run_search
from plain_memory.text import encoding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--since", type=datetime.fromisoformat, required=True)
    parser.add_argument("--until", type=datetime.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    records, malformed = load_records(args.journal)
    settings = load_settings(args.config)
    if settings.retrieval.rerank or settings.retrieval.query_expansion:
        raise SystemExit(
            "Replay requires rerank/query_expansion disabled; do not silently change them."
        )
    settings.embedding.cache_path = args.cache
    encoder = HTTPEmbedder(settings.embedding)
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with sqlite3.connect(settings.database.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            for record in records:
                when = datetime.fromisoformat(record["time"])
                if record["endpoint"] != "search" or not args.since <= when <= args.until:
                    continue
                request = SearchRequest.model_validate(record["request"])
                allowed = {r["request"]["request_id"] for r in prior_adds(records, record)}
                # Filter before building ANY retrieval structure: future facts,
                # updates and graded attempts must not enter the earlier query.
                rows = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM chunks WHERE user_id=? ORDER BY rowid", (request.user_id,)
                    )
                    if row["request_id"] in allowed
                ]
                history = hashlib.sha256()
                for row in rows:
                    history.update(
                        json.dumps(
                            {k: v for k, v in row.items() if k != "embedding"},
                            sort_keys=True,
                        ).encode()
                    )
                    history.update(row["embedding"])
                started = time.perf_counter()
                index = UserIndex(rows) if rows else None
                items = (
                    run_search(
                        index, request, settings, encoder, None, None, time.monotonic() + 120
                    )
                    if index
                    else []
                )
                token_count = sum(
                    len(encoding().encode(i["content"], disallowed_special=())) for i in items
                )
                case_id = hashlib.sha256((request.user_id + record["time"]).encode()).hexdigest()[
                    :16
                ]
                results.append(
                    {
                        "case_id": case_id,
                        "time": record["time"],
                        "question": request.query,
                        "items": items,
                        "tokens": token_count,
                        "eligible_adds": len(allowed),
                        "eligible_rows": len(rows),
                        "history_sha256": history.hexdigest(),
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                )
                print(
                    json.dumps(
                        {
                            "completed": len(results),
                            "case_id": case_id,
                            "items": len(items),
                            "tokens": token_count,
                        }
                    ),
                    flush=True,
                )
    finally:
        encoder.close()
    with args.out.open("w", encoding="utf-8") as stream:
        args.out.chmod(0o600)
        json.dump(
            {"malformed_lines": malformed, "cases": results}, stream, ensure_ascii=False, indent=2
        )


if __name__ == "__main__":
    main()

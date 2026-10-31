"""Audit a request journal on the machine holding it; never infer missing gold labels.

Example: python eval/audit_journal.py --journal data/journal/requests.jsonl
         --since 2026-10-29T03:04:17+00:00 --until 2026-10-29T03:43:06+00:00
         --out data/analysis/journal-audit.json
"""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from plain_memory import feedback, intent, timeparse
from plain_memory.text import encoding


def load_records(path):
    records, malformed = [], 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                datetime.fromisoformat(record["time"])
                if not isinstance(record.get("request"), dict):
                    raise ValueError
                records.append(record)
            except (ValueError, KeyError, TypeError):
                malformed += 1
    return sorted(records, key=lambda r: datetime.fromisoformat(r["time"])), malformed


def prior_adds(records, search):
    """Conservative replay: only Adds completed before Search started are eligible."""
    start = datetime.fromisoformat(search["time"]) - timedelta(milliseconds=search["elapsed_ms"])
    user = search["request"]["user_id"]
    found = {}
    for record in records:
        if datetime.fromisoformat(record["time"]) > start:
            break
        request = record["request"]
        if (
            record["endpoint"] == "add"
            and record["status"] == 200
            and request.get("user_id") == user
        ):
            found.setdefault(request["request_id"], record)
    return list(found.values())


def audit(records, since=None, until=None):
    from plain_memory.searching import query_with_options
    from plain_memory.state import parse_fact

    selected = [
        r
        for r in records
        if (since is None or datetime.fromisoformat(r["time"]) >= since)
        and (until is None or datetime.fromisoformat(r["time"]) <= until)
    ]
    cases = []
    for record in selected:
        if record["endpoint"] != "search":
            continue
        request = record["request"]
        messages = [m for r in prior_adds(records, record) for m in r["request"]["messages"]]
        items = (record.get("response") or {}).get("data", [])
        tokens = sum(len(encoding().encode(i["content"], disallowed_special=())) for i in items)
        old_query = request["query"] + (
            "\nOptions:\n" + "\n".join(request["options"]) if request.get("options") else ""
        )
        cleaned = query_with_options(request["query"], request.get("options"))
        flags = []
        if not messages:
            flags.append("no_prior_history")
        elif not items:
            flags.append("empty_despite_history")
        if cleaned != old_query:
            flags.append("duplicated_query_options")
        if tokens > 16000:
            flags.append("long_returned_context")
        embedded_times = sum(
            m.get("timestamp") is None and timeparse.message_timestamp(m["content"]) is not None
            for m in messages
        )
        if embedded_times:
            flags.append("timestamps_only_in_message_text")
        replacements = [parse_fact(m["content"]) for m in messages if m["role"] == "user"]
        if any(f and f.replace for f in replacements):
            flags.append("explicit_fact_replacements_available")
        cases.append(
            {
                "case_id": hashlib.sha256(
                    (request["user_id"] + record["time"]).encode()
                ).hexdigest()[:16],
                "time": record["time"],
                "status": record["status"],
                "user_hash": hashlib.sha256(request["user_id"].encode()).hexdigest()[:12],
                "question": intent.redact_secrets(request["query"], ""),
                "items": len(items),
                "tokens": tokens,
                "elapsed_ms": record["elapsed_ms"],
                "prior_messages": len(messages),
                "embedded_timestamps": embedded_times,
                "query_tokens_before": len(encoding().encode(old_query, disallowed_special=())),
                "query_tokens_after": len(encoding().encode(cleaned, disallowed_special=())),
                "flags": flags,
                "grade": None,
            }
        )
    searches = [r for r in selected if r["endpoint"] == "search"]
    seen_feedback = set()
    for record in selected:
        if record["endpoint"] != "add" or record["status"] != 200:
            continue
        request = record["request"]
        for number, message in enumerate(request.get("messages", [])):
            key = request["user_id"], request["request_id"], number
            if key in seen_feedback:
                continue
            seen_feedback.add(key)
            attempt = feedback.parse(message["content"], message["role"])
            if attempt is None:
                continue
            for i in range(len(searches) - 1, -1, -1):
                previous = searches[i]
                if (
                    previous["request"]["user_id"] == request["user_id"]
                    and previous["request"]["query"].strip() == attempt.question
                    and datetime.fromisoformat(previous["time"])
                    <= datetime.fromisoformat(record["time"])
                ):
                    cases[i]["grade"] = {
                        "score": attempt.score,
                        "source": "online_feedback",
                        "predicted_answer": intent.redact_secrets(attempt.answer, ""),
                    }
                    break
    labelled = [c for c in cases if c["grade"] is not None]
    return {
        "summary": {
            "records": len(selected),
            "search_calls": len(cases),
            "search_statuses": dict(Counter(str(c["status"]) for c in cases)),
            "labelled_calls": len(labelled),
            "unlabelled_calls": len(cases) - len(labelled),
            "labelled_incorrect": sum(c["grade"]["score"] == 0 for c in labelled),
            "labelled_partial": sum(0 < c["grade"]["score"] < 1 for c in labelled),
            "flags": dict(Counter(flag for c in cases for flag in c["flags"])),
            "median_search_ms": statistics.median(c["elapsed_ms"] for c in cases)
            if cases
            else None,
            "max_returned_tokens": max((c["tokens"] for c in cases), default=0),
        },
        "limitations": [
            "Search calls are not necessarily distinct scored questions.",
            "Flags diagnose evidence handling, not answer correctness.",
            "No answer labels are inferred for calls without online feedback.",
            "Concurrent Adds finishing after Search starts are excluded from replay.",
        ],
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--since", type=datetime.fromisoformat)
    parser.add_argument("--until", type=datetime.fromisoformat)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records, malformed = load_records(args.journal)
    report = audit(records, args.since, args.until)
    report["summary"]["malformed_lines"] = malformed
    args.out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.out.open("w", encoding="utf-8") as stream:
        args.out.chmod(0o600)
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=True))


if __name__ == "__main__":
    main()

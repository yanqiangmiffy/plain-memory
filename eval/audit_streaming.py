"""Count how streaming (online-feedback) users are served; prints numbers only, no question text.

Run it on the machine that holds the request journal:

    python eval/audit_streaming.py --journal data/journal/requests.jsonl

A "run" is a burst of Search calls with no gap longer than 30 minutes. A user counts as
streaming when one of their Adds carries a "Streaming online QA feedback:" envelope. Per run it
reports how many Searches were exact repeats of a question that had already been graded, how many
responses contained a graded-attempt excerpt, and how large the responses were.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PREFIX = "Streaming online QA feedback:"


def norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def load(path: Path) -> list[dict]:
    records = []
    for line in path.open(encoding="utf-8"):
        try:
            record = json.loads(line)
            record["_t"] = datetime.fromisoformat(record["time"])
            records.append(record)
        except (ValueError, KeyError, TypeError):
            continue
    return sorted(records, key=lambda r: r["_t"])


def split_runs(searches: list[dict], gap_seconds: int = 1800) -> list[list[dict]]:
    runs: list[list[dict]] = []
    for search in searches:
        if runs and (search["_t"] - runs[-1][-1]["_t"]).total_seconds() <= gap_seconds:
            runs[-1].append(search)
        else:
            runs.append([search])
    return runs


def graded_attempts(adds: list[dict]) -> dict[str, list[tuple]]:
    found = defaultdict(list)
    for add in adds:
        user = add["request"]["user_id"]
        for message in add["request"].get("messages", []):
            content = message.get("content", "")
            if message.get("role") != "user" or not content.startswith(PREFIX):
                continue
            try:
                value = json.loads(content[len(PREFIX) :].strip())
                found[user].append((add["_t"], norm(value["question"]), value.get("score")))
            except (ValueError, KeyError, TypeError):
                continue
    return found


def describe(label: str, group: list[dict]) -> None:
    if not group:
        print(f"  {label}: none")
        return
    sizes = [sum(len(i["content"]) for i in (s.get("response") or {}).get("data", [])) / 4 for s in group]
    items = [len((s.get("response") or {}).get("data", [])) for s in group]
    print(
        f"  {label}: searches={len(group)} users={len({s['request']['user_id'] for s in group})} "
        f"with_options={sum(bool(s['request'].get('options')) for s in group)} "
        f"mean_items={statistics.mean(items):.1f} approx_tokens={statistics.mean(sizes):.0f} "
        f"median_ms={statistics.median(s['elapsed_ms'] for s in group):.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    records = load(args.journal)
    print("records", len(records), dict(Counter((r["endpoint"], r["status"]) for r in records)))
    searches = [r for r in records if r["endpoint"] == "search"]
    for run in split_runs(searches):
        last = run[-1]["_t"]
        adds = [r for r in records if r["endpoint"] == "add" and r["status"] == 200 and r["_t"] <= last]
        attempts = graded_attempts(adds)
        streaming = [s for s in run if s["request"]["user_id"] in attempts]
        ordinary = [s for s in run if s["request"]["user_id"] not in attempts]
        print(f"RUN {run[0]['_t']:%m-%d %H:%M}..{last:%H:%M} searches={len(run)}")
        describe("streaming", streaming)
        describe("textual", ordinary)
        repeats = passed = failed = shown = 0
        per_user = defaultdict(Counter)
        for s in streaming:
            user, query = s["request"]["user_id"], norm(s["request"]["query"])
            per_user[user][query] += 1
            earlier = [a for a in attempts[user] if a[0] <= s["_t"] and a[1] == query]
            if earlier:
                repeats += 1
                passed += any((a[2] or 0) >= 1 for a in earlier)
                failed += all((a[2] or 0) <= 0 for a in earlier)
            data = (s.get("response") or {}).get("data", [])
            shown += any("graded" in item["content"][:120] for item in data)
        if streaming:
            same = sum(sum(c.values()) - len(c) for c in per_user.values())
            scores = Counter(a[2] for v in attempts.values() for a in v)
            print(
                f"  streaming: exact repeats of a graded question={repeats} (passed={passed}, "
                f"only failed={failed}); responses with a graded excerpt={shown}; "
                f"identical queries repeated within a user={same}"
            )
            print(f"  graded attempts: users={len(attempts)} scores={dict(scores)}")


if __name__ == "__main__":
    main()

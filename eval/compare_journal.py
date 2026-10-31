"""Compare retrieval replays; evidence changes must never be reported as accuracy gains."""

import argparse
import json
import statistics
from pathlib import Path


def compare(baseline, candidate):
    before = {case["case_id"]: case for case in baseline["cases"]}
    after = {case["case_id"]: case for case in candidate["cases"]}
    if len(before) != len(baseline["cases"]) or len(after) != len(candidate["cases"]):
        raise ValueError("Duplicate case IDs in replay.")
    if before.keys() != after.keys():
        raise ValueError("Replays contain different cases.")
    cases = []
    for case_id, old in before.items():
        new = after[case_id]
        if (
            not old.get("history_sha256")
            or old["history_sha256"] != new.get("history_sha256")
            or old["question"] != new["question"]
        ):
            raise ValueError("Replays must use identical historical inputs and questions.")
        old_items = {item["id"]: item for item in old["items"]}
        # Match the same excerpt ID, not its position: new summary cards and
        # reranking can otherwise masquerade as a timestamp repair.
        time_changes = sum(
            item["id"] in old_items
            and item["created_at"] != old_items[item["id"]]["created_at"]
            for item in new["items"]
        )
        cases.append(
            {
                "case_id": case_id,
                "response_changed": old["items"] != new["items"],
                "state_view": any(
                    item["content"].startswith("[Structured evidence") for item in new["items"]
                ),
                "historical_fact_marked": any(
                    "[Historical fact;" in item["content"] for item in new["items"]
                ),
                "same_excerpt_timestamp_changes": time_changes,
                "baseline_tokens": old["tokens"],
                "candidate_tokens": new["tokens"],
            }
        )
    return {
        "summary": {
            "replayed": len(cases),
            "identical_historical_inputs": True,
            "changed_responses": sum(c["response_changed"] for c in cases),
            "state_views": sum(c["state_view"] for c in cases),
            "old_facts_marked": sum(c["historical_fact_marked"] for c in cases),
            "queries_with_same_excerpt_timestamp_changes": sum(
                c["same_excerpt_timestamp_changes"] > 0 for c in cases
            ),
            "mean_baseline_tokens": statistics.mean(c["baseline_tokens"] for c in cases)
            if cases
            else None,
            "mean_candidate_tokens": statistics.mean(c["candidate_tokens"] for c in cases)
            if cases
            else None,
        },
        "limitations": [
            "Retrieval-only replay: no answer model or new accuracy score.",
            "Changes in evidence are diagnostic, not proof of better answers.",
        ],
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.out.open("w", encoding="utf-8") as stream:
        args.out.chmod(0o600)
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()

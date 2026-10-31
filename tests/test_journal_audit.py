import json
from datetime import datetime, timezone

import pytest

from eval.audit_journal import audit, prior_adds
from eval.compare_journal import compare


def record(endpoint, seconds, request, response=None, elapsed_ms=0):
    return {
        "time": datetime(2026, 1, 1, 0, 0, seconds, tzinfo=timezone.utc).isoformat(),
        "endpoint": endpoint,
        "request": request,
        "response": response,
        "status": 200,
        "elapsed_ms": elapsed_ms,
    }


def add(request_id, text, user="u"):
    return {
        "user_id": user,
        "request_id": request_id,
        "messages": [{"role": "user", "content": text}],
    }


def test_audit_does_not_treat_missing_labels_as_incorrect_or_replay_future_adds():
    search = record("search", 10, {"user_id": "u", "query": "Q?", "top_k": 100}, {"data": []}, 2000)
    records = [
        record("add", 1, add("old", "an old fact")),
        record("add", 2, add("old", "an old fact")),  # idempotent retry
        record("add", 3, add("other", "another user's fact", "other")),
        record("add", 9, add("concurrent", "completed after search started")),
        search,
        record("add", 11, add("future", "future update")),
    ]
    assert [r["request"]["request_id"] for r in prior_adds(records, search)] == ["old"]
    report = audit(records)
    assert report["summary"]["unlabelled_calls"] == 1
    assert report["summary"]["labelled_incorrect"] == 0
    assert report["cases"][0]["prior_messages"] == 1


def test_feedback_labels_only_latest_matching_prior_query_and_deduplicates_retries():
    payload = add(
        "grade",
        "Streaming online QA feedback:\n"
        + json.dumps(
            {
                "question": "Q?",
                "predicted_answer": "wrong",
                "score": 0,
            }
        ),
    )
    records = [
        record("search", 1, {"user_id": "u", "query": "Q?"}, {"data": []}),
        record("search", 2, {"user_id": "u", "query": "Q?"}, {"data": []}),
        record("add", 3, payload),
        record("add", 4, payload),
        record("search", 5, {"user_id": "u", "query": "Q?"}, {"data": []}),
    ]
    report = audit(records)
    assert report["summary"]["labelled_incorrect"] == 1
    assert [c["grade"] is not None for c in report["cases"]] == [False, True, False]


def test_replay_comparison_matches_excerpt_ids_and_rejects_changed_history():
    case = {
        "case_id": "c1",
        "question": "Q?",
        "history_sha256": "same-history",
        "tokens": 10,
        "items": [
            {"id": "first", "content": "A", "created_at": "2026-01-01"},
            {"id": "second", "content": "B", "created_at": "2026-01-02"},
        ],
    }
    baseline = {"cases": [case]}
    candidate = {"cases": [{**case, "items": list(reversed(case["items"]))}]}
    result = compare(baseline, candidate)["summary"]
    assert result["changed_responses"] == 1
    assert result["queries_with_same_excerpt_timestamp_changes"] == 0
    candidate["cases"][0]["items"] = [
        {**case["items"][0], "created_at": "2025-01-01"}, case["items"][1]
    ]
    result = compare(baseline, candidate)["summary"]
    assert result["queries_with_same_excerpt_timestamp_changes"] == 1
    candidate["cases"][0]["history_sha256"] = "future-facts-leaked-in"
    with pytest.raises(ValueError, match="identical historical inputs"):
        compare(baseline, candidate)

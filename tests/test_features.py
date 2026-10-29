import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from plain_memory import feedback, intent, timeparse
from plain_memory.api import create_app

FRIDAY = datetime(2023, 6, 9, 13, 56, tzinfo=timezone.utc)
STAMP = int(FRIDAY.timestamp() * 1000)


def test_relative_notes_keep_the_granularity_of_the_phrase():
    notes = timeparse.relative_notes("I saw her yesterday and last week and last year", FRIDAY)
    assert "yesterday = 8 June 2023" in notes
    assert "last week = the week before 9 June 2023 (29 May - 4 June 2023)" in notes
    assert "last year = 2022" in notes
    assert timeparse.relative_notes("last Friday", FRIDAY) == ["last friday = 2 June 2023"]
    assert timeparse.relative_notes("nothing relative here", FRIDAY) == []


def test_query_dates_ignore_ranges_and_bare_years():
    day = timeparse.query_dates("What happened on 25 May, 2022?")
    assert day and day[0][0].isoformat() == "2022-05-25"
    month = timeparse.query_dates("events in May 2023")
    assert month[0][0].isoformat() == "2023-05-01" and month[0][1].isoformat() == "2023-05-31"
    assert timeparse.query_dates("before 8 May 2023") == []
    assert timeparse.query_dates("in 1999") == []


def test_feedback_envelope_is_strict():
    body = (
        'Streaming online QA feedback:\n{"qa_id":"q1","question":"Q?",'
        '"predicted_answer":"A","score":1}'
    )
    attempt = feedback.parse(body, "user")
    assert attempt and attempt.outcome == "passed" and "CORRECT" in attempt.render()
    assert feedback.parse(body, "assistant") is None
    assert feedback.parse(body.replace('"score":1', '"score":2'), "user") is None
    assert feedback.parse("Streaming online QA feedback: not json", "user") is None
    failed = feedback.parse(body.replace('"score":1', '"score":0'), "user")
    assert failed and failed.outcome == "failed"


def test_intents_and_redaction():
    assert intent.wants_current_state("Where do I currently live?")
    assert not intent.wants_current_state("Where did I live in 2019?")
    assert intent.wants_recommendation("Can you recommend a good book?")
    assert intent.wants_overview("Please summarize our entire history")
    text = "my password is hunter2xyz and key sk-abcdefghijklmnop1234"
    masked = intent.redact_secrets(text, "What did we eat?")
    assert "hunter2xyz" not in masked and "sk-abcdef" not in masked
    assert intent.redact_secrets(text, "What is my password?") == text
    assert intent.first_person_sentences("I love hiking in Denver. Do you like it?") == [
        "I love hiking in Denver."
    ]


def _client(settings, fake, **retrieval):
    for key, value in retrieval.items():
        setattr(settings.retrieval, key, value)
    return TestClient(create_app(settings, embedder=fake))


def _add(client, headers, request, messages, user="u"):
    payload = {
        "request_id": request,
        "user_id": user,
        "session_id": "s-" + request,
        "messages": messages,
    }
    assert client.post("/add", headers=headers, json=payload).status_code == 200


def _search(client, headers, query, user="u"):
    response = client.post(
        "/search", headers=headers, json={"query": query, "user_id": user, "top_k": 100}
    )
    assert response.status_code == 200
    return response.json()["data"]


def test_rule_notes_are_stored_and_shown(settings, fake, headers):
    with _client(settings, fake, rule_notes=True) as client:
        _add(
            client,
            headers,
            "r1",
            [{"role": "user", "content": "I went hiking yesterday", "timestamp": STAMP}],
        )
        text = _search(client, headers, "hiking")[0]["content"]
        assert "(time note: yesterday = 8 June 2023)" in text


def test_date_boost_pulls_in_the_queried_day(settings, fake, headers):
    day = lambda d: int(datetime(2023, 6, d, 12, tzinfo=timezone.utc).timestamp() * 1000)  # noqa: E731
    with _client(
        settings, fake, date_boost=0.5, candidate_units=1, window_before=0, window_after=0
    ) as client:
        _add(
            client,
            headers,
            "a",
            [{"role": "user", "content": "zebra zebra zebra", "timestamp": day(1)}],
        )
        _add(
            client,
            headers,
            "b",
            [{"role": "user", "content": "unrelated words only", "timestamp": day(5)}],
        )
        data = _search(client, headers, "zebra on 5 June 2023")
        assert "unrelated words only" in data[0]["content"]


def test_feedback_attempts_are_labelled_and_failed_ones_demoted(settings, fake, headers):
    def envelope(question, answer, score):
        value = {"qa_id": "x", "question": question, "predicted_answer": answer, "score": score}
        return "Streaming online QA feedback:\n" + json.dumps(value)

    with _client(settings, fake, window_before=0, window_after=0) as client:
        _add(
            client,
            headers,
            "good",
            [{"role": "user", "content": envelope("capital of Peru orchid", "Lima", 1)}],
        )
        _add(
            client,
            headers,
            "bad",
            [{"role": "user", "content": envelope("capital of Peru orchid", "Cusco", 0)}],
        )
        data = _search(client, headers, "capital of Peru orchid")
        assert "CORRECT" in data[0]["content"] and "Lima" in data[0]["content"]
        assert "INCORRECT" in data[-1]["content"]


def test_current_state_newest_first_and_overview_outline(settings, fake, headers):
    stamps = [int(datetime(2023, m, 1, tzinfo=timezone.utc).timestamp() * 1000) for m in (1, 2, 3)]
    with _client(
        settings, fake, current_state_order=True, overview_outline=True, profile_digest=True
    ) as client:
        for i, ts in enumerate(stamps):
            _add(
                client,
                headers,
                f"r{i}",
                [{"role": "user", "content": f"I live in city{i} orchid", "timestamp": ts}],
            )
        data = _search(client, headers, "Where do I currently live orchid?")
        order = [d["created_at"] for d in data]
        assert order == sorted(order, reverse=True)
        overview = _search(client, headers, "Please summarize our entire conversation history")
        assert overview[0]["content"].startswith("[Overview of the whole history")
        assert overview[0]["content"].count("\n- ") == 3
        profile = _search(client, headers, "Can you recommend something for me, orchid?")
        assert profile[0]["content"].startswith("[User profile notes")


def test_secrets_are_masked_in_results(settings, fake, headers):
    with _client(settings, fake) as client:
        _add(
            client,
            headers,
            "r",
            [{"role": "user", "content": "my password is hunter2xyz ok", "timestamp": STAMP}],
        )
        assert "hunter2xyz" not in _search(client, headers, "what did I say ok")[0]["content"]
        assert "hunter2xyz" in _search(client, headers, "what is my password")[0]["content"]


def test_request_journal_records_calls_and_failures(settings, fake, headers, tmp_path):
    settings.request_journal = tmp_path / "journal" / "requests.jsonl"
    with TestClient(create_app(settings, embedder=fake)) as client:
        _add(
            client, headers, "r1", [{"role": "user", "content": "orchid memo", "timestamp": STAMP}]
        )
        _search(client, headers, "orchid")
        bad = client.post(
            "/search", headers=headers, json={"query": "  ", "user_id": "u", "top_k": 5}
        )
        assert bad.status_code == 422
    lines = [json.loads(line) for line in settings.request_journal.read_text("utf-8").splitlines()]
    assert [(r["endpoint"], r["status"]) for r in lines] == [
        ("add", 200),
        ("search", 200),
        ("search", 422),
    ]
    assert lines[0]["request"]["messages"][0]["content"] == "orchid memo"
    assert "orchid memo" in lines[1]["response"]["data"][0]["content"]
    assert lines[2]["error"] and lines[2]["response"] is None
    assert settings.api_token.get_secret_value() not in settings.request_journal.read_text("utf-8")
    if hasattr(settings.request_journal, "stat") and not __import__("os").name == "nt":
        assert oct(settings.request_journal.stat().st_mode & 0o777) == "0o600"


def test_request_journal_stops_at_its_size_limit(settings, fake, headers, tmp_path):
    settings.request_journal = tmp_path / "j.jsonl"
    settings.journal_max_bytes = 1024
    with TestClient(create_app(settings, embedder=fake)) as client:
        for i in range(6):
            _add(
                client,
                headers,
                f"r{i}",
                [{"role": "user", "content": "x" * 400, "timestamp": STAMP}],
            )
    assert settings.request_journal.stat().st_size <= 1024

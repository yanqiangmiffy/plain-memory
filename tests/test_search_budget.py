import time
from types import SimpleNamespace

import pytest

from plain_memory import searching


@pytest.fixture
def run(monkeypatch, settings):
    settings.retrieval.context_tokens = 1000
    settings.retrieval.window_before = settings.retrieval.window_after = 0
    settings.retrieval.structured_state = False

    def invoke(texts, *, top_k=100, query="orchid"):
        items = [
            {"id": str(i), "content": text, "created_at": "2024-01-01T00:00:00Z"}
            for i, text in enumerate(texts)
        ]
        monkeypatch.setattr(searching, "fuse", lambda *a, **kw: {})
        monkeypatch.setattr(
            searching,
            "build_blocks",
            lambda *a: [(1 / (i + 1), str(i), 0, 0) for i in range(len(texts))],
        )
        monkeypatch.setattr(searching, "render_block", lambda idx, key, *a: items[int(key)])
        encoder = SimpleNamespace(embed=lambda texts, deadline: [[1.0, 0.0] for _ in texts])
        request = SimpleNamespace(query=query, options=None, top_k=top_k)
        return searching.run_search(
            None, request, settings, encoder, None, None, time.monotonic() + 10
        )

    return invoke


def test_skip_large_middle_candidate_and_use_later_evidence(run):
    data = run([" alpha" * 600, " beta" * 600, " gamma" * 300], top_k=2)
    assert [item["id"] for item in data] == ["0", "2"]
    assert sum(searching._tokens(item["content"]) for item in data) <= 1000


def test_first_oversized_excerpt_is_bounded_and_marked(run):
    data = run(["中文👩🏽‍💻 alpha " * 1000])
    assert len(data) == 1
    assert searching._tokens(data[0]["content"]) <= 1000
    assert "Excerpt truncated" in data[0]["content"]
    assert "\ufffd" not in data[0]["content"]


def test_profile_digest_counts_toward_token_and_item_budgets(run, monkeypatch, settings):
    settings.retrieval.profile_digest = True
    settings.retrieval.max_blocks = 2
    monkeypatch.setattr(
        searching,
        "profile_item",
        lambda *a: {
            "id": "profile",
            "content": " profile" * 800,
            "created_at": "2024-01-01T00:00:00Z",
        },
    )
    data = run([" alpha" * 600, " beta" * 80], query="Can you recommend a book?")
    assert [item["id"] for item in data] == ["profile", "0"]
    assert sum(searching._tokens(item["content"]) for item in data) <= 1000
    assert searching._tokens(data[0]["content"]) <= 250


def test_multiple_summaries_share_budget_and_leave_room_for_source_evidence(
    run, monkeypatch, settings
):
    settings.retrieval.structured_state = True
    settings.retrieval.profile_digest = True
    for function, name in (("structured_item", "state"), ("profile_item", "profile")):
        monkeypatch.setattr(
            searching,
            function,
            lambda *a, name=name: {
                "id": name,
                "content": (" " + name) * 180,
                "created_at": "2024-01-01T00:00:00Z",
            },
        )
    data = run([" alpha" * 700], query="Can you recommend a book?")
    assert [item["id"] for item in data] == ["state", "profile", "0"]
    assert sum(searching._tokens(item["content"]) for item in data[:2]) <= 250
    assert sum(searching._tokens(item["content"]) for item in data) <= 1000
    assert "truncated" not in data[-1]["content"]

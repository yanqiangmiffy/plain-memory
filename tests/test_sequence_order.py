import time
from types import SimpleNamespace

import pytest

from plain_memory import searching

DATES = {"0": "2024-03-01T00:00:00Z", "1": "2024-01-01T00:00:00Z", "2": "2024-02-01T00:00:00Z"}


@pytest.fixture
def ask(monkeypatch, settings):
    settings.retrieval.window_before = settings.retrieval.window_after = 0
    settings.retrieval.structured_state = False
    items = {
        key: {"id": key, "content": f"event {key}", "created_at": when}
        for key, when in DATES.items()
    }
    monkeypatch.setattr(searching, "fuse", lambda *a, **kw: {})
    monkeypatch.setattr(
        searching,
        "build_blocks",
        lambda *a: [(1 / (i + 1), key, 0, 0) for i, key in enumerate(DATES)],
    )
    monkeypatch.setattr(searching, "render_block", lambda idx, key, *a: items[key])
    encoder = SimpleNamespace(embed=lambda texts, deadline: [[1.0, 0.0] for _ in texts])

    def invoke(query):
        request = SimpleNamespace(query=query, options=None, top_k=100)
        data = searching.run_search(
            None, request, settings, encoder, None, None, time.monotonic() + 10
        )
        return [item["id"] for item in data]

    return invoke


def test_sequence_questions_return_the_oldest_evidence_first_when_enabled(ask, settings):
    settings.retrieval.sequence_order = True
    assert ask("Which happened first, the move or the job?") == ["1", "2", "0"]


def test_ordinary_questions_keep_relevance_order(ask, settings):
    settings.retrieval.sequence_order = True
    assert ask("Where does she live?") == ["0", "1", "2"]


def test_the_ordering_is_off_unless_configured(ask):
    assert ask("Which happened first, the move or the job?") == ["0", "1", "2"]

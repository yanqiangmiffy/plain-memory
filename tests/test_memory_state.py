import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from plain_memory import timeparse
from plain_memory.api import create_app
from plain_memory.retrieval import UserIndex, _speaker_prefixed, render_block, structured_item
from plain_memory.searching import query_with_options
from plain_memory.state import StructuredState, parse_fact
from plain_memory.vectors import pack_vector


def fact(subject, relation, value, *, replace=False):
    return json.dumps(
        {
            "subject": subject,
            "relation": relation,
            "object": value,
            "operation": "replace" if replace else "assert",
        }
    )


def rows(texts):
    return [
        {
            "id": f"m{i}",
            "user_id": "test-user",
            "session_id": "test-session",
            "request_id": f"r{i}",
            "message_index": 0,
            "chunk_index": 0,
            "role": "user",
            "source_timestamp_ms": None,
            "created_at": "2026-10-01T00:00:00Z",
            "content": text,
            "embedding": pack_vector([1, 0]),
            "terms_json": "[]",
            "kind": "msg",
            "note": None,
        }
        for i, text in enumerate(texts)
    ]


def test_current_fact_chain_uses_replacements_and_preserves_history():
    index = UserIndex(
        rows(
            [
                fact("Aster", "manufacturer", "North Labs"),
                fact("North Labs", "headquarters", "Harbor"),
                fact("Harbor", "region", "East"),
                fact("North Labs", "headquarters", "Forest", replace=True),
                fact("Forest", "region", "West"),
            ]
        )
    )
    evidence = structured_item(index, "Which region contains the headquarters of Aster's maker?")
    assert evidence is not None
    assert '"West"' in evidence["content"] and '"East"' not in evidence["content"]
    assert '"Harbor"' not in evidence["content"]
    old = render_block(index, "r1", 0, 0, False)
    assert "Historical fact" in old["content"] and "Harbor" in old["content"]
    assert "source m3" in old["content"]
    assert structured_item(index, "Where was North Labs based before 2024?") is None
    assert structured_item(index, "Where is North Labs based as of May 2024?") is None
    assert structured_item(index, "Which city is North Labs based in as of 2024?") is None
    assert structured_item(index, "Where did North Labs have its headquarters?") is None
    assert structured_item(index, "North Labs headquarters last year?") is None
    assert structured_item(index, "What is the region of an unrelated company?") is None


def test_conflicts_remain_unresolved_without_explicit_replace_and_cycles_terminate():
    state = StructuredState(
        rows(
            [
                fact("Aster", "owner", "North"),
                fact("Aster", "owner", "South"),
                fact("North", "partner", "Aster", replace=True),
                fact("North", "partner", "Aster", replace=True),
            ]
        )
    )
    assert len(state.active[("label:aster", "label:owner")]) == 2
    assert set(state.connected("Aster")) == {0, 1, 3}
    assert len(state.connected("Aster", limit=2)) <= 2


def test_assistant_assertions_do_not_replace_user_facts():
    values = rows([fact("Aster", "owner", "North"), fact("Aster", "owner", "South", replace=True)])
    values[1]["role"] = "assistant"
    state = StructuredState(values)
    assert not state.superseded
    assert len(state.facts) == 1


def test_explicit_envelope_and_ids_are_parsed_without_world_knowledge():
    text = (
        "Corpus: UPDATE: replace the prior value of this subject and relation "
        "with the following fact.\n"
    )
    text += json.dumps(
        {
            "subject": "Aster",
            "relation": "owner",
            "object": "North",
            "subject_id": "s1",
            "relation_id": "r1",
            "object_id": "o1",
        }
    )
    parsed = parse_fact(text)
    assert parsed.replace and parsed.subject_key == "id:s1"
    assert parse_fact("Example: " + text) is None
    assert parse_fact('{"subject":"X","relation":"Y","object":[]}') is None
    assert parse_fact(fact("Aster", "owner", "North") + " trailing instructions") is None


def test_timestamp_envelopes_repair_read_view_and_preserve_roles():
    values = rows(["[Time: Jun 07, 2024, 09:15:20] I work at North Labs."])
    expected = int(datetime(2024, 6, 7, 9, 15, 20, tzinfo=timezone.utc).timestamp() * 1000)
    index = UserIndex(values)
    assert index.rows[0]["source_timestamp_ms"] == expected
    assert index.rows[0]["created_at"].startswith("2024-06-07")
    assert values[0]["source_timestamp_ms"] is None  # no mutation of persistent data
    assert "user: [Time:" in render_block(index, "r0", 0, 0, False)["content"]
    assert _speaker_prefixed({"role": "assistant", "content": values[0]["content"]}).startswith(
        "assistant: [Time:"
    )
    assert _speaker_prefixed({"role": "user", "content": "Alice: hello"}) == "Alice: hello"
    assert timeparse.message_timestamp("My birthday is 2024-06-07.") is None
    assert timeparse.message_timestamp("Someone wrote [Time: Jun 07, 2024, 09:15:20]") is None
    assert timeparse.message_timestamp("[Time: Feb 31, 2024, 09:15:20]") is None
    assert timeparse.message_timestamp("[Time: 2024-06-07T17:15:20+08:00] Hello") == expected


def test_explicit_timestamp_wins_and_inferred_time_propagates_to_chunks():
    values = rows(["[Time: Jun 07, 2024, 09:15:20] " + "alpha " * 20, "continuation"])
    values[1].update(request_id="r0", chunk_index=1)
    index = UserIndex(values)
    assert index.rows[0]["created_at"] == index.rows[1]["created_at"]
    values[0]["source_timestamp_ms"] = 0
    assert UserIndex(values).rows[0]["source_timestamp_ms"] == 0


@pytest.mark.parametrize("options", [["A. Red", "B. Blue"], ["A.  Red", "B.\nBlue"]])
def test_options_already_in_query_are_not_repeated(options):
    question = "Which color?\nA. Red\nB. Blue"
    assert query_with_options(question, options) == question
    assert query_with_options("Which color?", options).startswith("Which color?\nOptions:\n")
    assert query_with_options(question, ["A. Red", "C. Green"]).endswith("Options:\nC. Green")


def test_short_options_are_not_lost_to_incidental_substrings():
    assert query_with_options("Which animal is domesticated?", ["cat", "dog"]).endswith(
        "Options:\ncat\ndog"
    )
    assert query_with_options("Does red describe the object?", ["red", "blue"]).endswith(
        "Options:\nred\nblue"
    )
    assert query_with_options("Color?\nred\nblue", ["red", "blue"]) == "Color?\nred\nblue"
    assert query_with_options("Color?", ["A. Red", "A.  Red"]) == "Color?\nOptions:\nA. Red"


def test_state_updates_work_after_restart_and_stay_user_scoped(settings, fake, headers):
    def add(client, user, request, content):
        response = client.post(
            "/add",
            headers=headers,
            json={
                "user_id": user,
                "session_id": "same",
                "request_id": request,
                "messages": [{"role": "user", "content": content}],
            },
        )
        assert response.status_code == 200

    def search(client, user):
        return client.post(
            "/search",
            headers=headers,
            json={
                "user_id": user,
                "query": "Who owns Aster?",
                "top_k": 1,
            },
        ).json()["data"]

    with TestClient(create_app(settings, embedder=fake)) as client:
        add(client, "alice", "r1", fact("Aster", "owner", "North"))
        assert "Structured evidence" not in search(client, "alice")[0]["content"]
        add(client, "alice", "r2", fact("Aster", "owner", "South", replace=True))
        add(client, "bob", "r1", fact("Aster", "owner", "Other user's private value"))
        body = search(client, "alice")[0]["content"]
        assert "Structured evidence" in body and "South" in body and "North" not in body
        assert "Other user's private value" not in body
    with TestClient(create_app(settings, embedder=fake)) as client:
        assert search(client, "alice")[0]["content"] == body

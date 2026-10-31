import json

import pytest

from plain_memory.api import create_app
from plain_memory.config import RetrievalConfig
from plain_memory.retrieval import UserIndex, build_blocks, fuse, interleave, render_block
from plain_memory.store import StoreMismatch
from plain_memory.text import chunks, encoding, lexical_terms
from plain_memory.vectors import pack_vector


def test_unicode_chunks_are_original_substrings_and_cover_message():
    original = "".join(f"中文👩🏽‍💻 café agent第{i}条记忆测试。" for i in range(20))
    fragments = chunks(original, 32, 4)
    covered = 0
    previous_start = -1
    for fragment in fragments:
        start = original.find(fragment, previous_start + 1)
        assert start >= 0 and start <= covered
        assert "\ufffd" not in fragment
        assert len(encoding().encode(fragment)) <= 32
        covered = max(covered, start + len(fragment))
        previous_start = start
    assert covered == len(original)
    assert chunks("short 中文", 512, 64) == ["short 中文"]
    assert chunks("literal <|endoftext|>", 512, 64) == ["literal <|endoftext|>"]


def test_no_overlap_covers_exact_original():
    text = "中英文 testing 🤖 " * 20
    assert "".join(chunks(text, 16, 0)) == text


def test_interleaving_keeps_first_occurrence_and_fills_from_remaining_route():
    assert interleave(["A", "B", "C", "D"], ["B", "E", "A", "F"], 4) == ["A", "B", "E", "C"]
    assert interleave(["A"], ["A", "B", "C"], 10) == ["A", "B", "C"]
    assert interleave([], [], 100) == []


def make_rows(texts, kind="msg", request="r1", note=None):
    rows = []
    for i, text in enumerate(texts):
        rows.append(
            {
                "id": f"{request}-{kind}-{i}",
                "user_id": "u",
                "request_id": request,
                "message_index": i,
                "chunk_index": 0,
                "role": "user",
                "source_timestamp_ms": 1704067200000,
                "created_at": "2024-01-01T00:00:00Z",
                "terms_json": json.dumps(lexical_terms(text)),
                "embedding": pack_vector([1.0] * 4),
                "kind": kind,
                "note": note if i == 0 else None,
                "content": text,
            }
        )
    return rows


def test_chinese_lexical_terms_and_stemmed_english():
    assert "北京" in lexical_terms("我住在北京")
    assert lexical_terms("The cats are running") == ["cat", "run"]


def test_blocks_expand_merge_and_render_with_date_and_notes():
    texts = [f"Alice: line {i}" for i in range(8)] + ["Bob: orchid garden"]
    rows = make_rows(texts, note="yesterday = 31 December 2023")
    index = UserIndex(rows)
    config = RetrievalConfig(window_before=1, window_after=1, candidate_units=1)
    query = [1.0] * 4
    scores = fuse(index, [("orchid", 1.0)], [query], config)
    blocks = build_blocks(index, scores, config)
    assert blocks[0][1:] == ("r1", 7, 8)  # orchid hit expands to its neighbour only
    item = render_block(index, "r1", 0, 1, True)
    assert item["content"].startswith("[Conversation excerpt | Monday, 1 January 2024]")
    assert "(time note: yesterday = 31 December 2023)" in item["content"]
    assert item["id"].startswith("mem_") and item["created_at"] == "2024-01-01T00:00:00Z"


def test_fact_hits_map_back_to_their_source_message():
    rows = make_rows(["Alice: hi", "Bob: bye", "Carol: ok"])
    fact = make_rows(["Alice adopted an orchid."], kind="fact")[0]
    fact["message_index"] = 2
    fact["embedding"] = pack_vector([-1.0] * 4)
    index = UserIndex(rows + [fact])
    config = RetrievalConfig(window_before=0, window_after=0, candidate_units=1)
    scores = fuse(index, [("orchid", 1.0)], [[1.0] * 4], config)
    (score, key, lo, hi), *_ = build_blocks(index, scores, config)
    assert (key, lo, hi) == ("r1", 2, 2)
    assert "Alice adopted an orchid." in render_block(index, key, lo, hi, True, True)["content"]
    assert "Facts:" not in render_block(index, key, lo, hi, True, False)["content"]


def test_model_switch_rejected_on_existing_database(settings, fake):
    app = create_app(settings, embedder=fake)
    app.state.store.health()
    settings.embedding.model = "text-embedding-v4"
    with pytest.raises(StoreMismatch):
        create_app(settings, embedder=fake)

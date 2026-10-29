"""The Search pipeline: query views -> fusion -> excerpts -> rerank -> selection -> final shape."""

import time

from plain_memory import intent
from plain_memory.config import RetrievalConfig
from plain_memory.embedding import EmbeddingError, normalize_vectors
from plain_memory.enrich import expand_query
from plain_memory.retrieval import (
    FB_KEY,
    UserIndex,
    build_blocks,
    fuse,
    outline_item,
    profile_item,
    render_block,
)
from plain_memory.text import encoding


def _tokens(text: str) -> int:
    return len(encoding().encode(text, disallowed_special=()))


def _session_of(index: UserIndex, order) -> str | None:
    key, lo = order
    if key == FB_KEY:
        return None
    return index.rows[index.groups[key][lo]]["session_id"]


def _round_robin(rendered: list, index: UserIndex) -> list:
    """Best excerpt of every session first, then the rest (keeps relevance order inside)."""
    seen, first, rest = set(), [], []
    for entry in rendered:
        session = _session_of(index, entry[2])
        if session is None or session in seen:
            rest.append(entry)
        else:
            seen.add(session)
            first.append(entry)
    return first + rest


def run_search(index, request, settings, encoder, llm, reranker, deadline: float) -> list[dict]:
    cfg: RetrievalConfig = settings.retrieval
    question = request.query
    full_query = question
    if request.options:
        full_query += "\nOptions:\n" + "\n".join(request.options)
    queries = [(full_query, 1.0)]
    if cfg.query_expansion and llm is not None:
        queries += [(t, 0.7) for t in expand_query(question, request.options, llm, deadline)]
    if cfg.option_views and request.options:
        queries += [(f"{question}\nCandidate answer: {o}", 0.5) for o in request.options[:8]]
    vectors = normalize_vectors(
        encoder.embed([text for text, _ in queries], deadline),
        len(queries),
        settings.embedding.dimensions,
    )
    scores = fuse(index, queries, vectors, cfg, original_query=question)
    blocks = build_blocks(index, scores, cfg)
    limit = min(request.top_k, cfg.max_blocks)
    rendered = [
        (score, render_block(index, key, lo, hi, cfg.show_notes, cfg.show_facts), (key, lo))
        for score, key, lo, hi in blocks
    ]
    if cfg.rerank and reranker is not None and rendered:
        head = rendered[: cfg.rerank_candidates]
        values = reranker.scores(
            full_query, [item["content"][: cfg.rerank_chars] for _, item, _ in head], deadline
        )
        if values is not None:
            top_rrf = max((score for score, _, _ in head), default=0.0) or 1.0
            top_rr = max(values) or 1.0
            mixed = [
                (
                    cfg.rerank_weight * v / top_rr + (1 - cfg.rerank_weight) * sc / top_rrf,
                    item,
                    order,
                )
                for (sc, item, order), v in zip(head, values)
            ]
            mixed.sort(key=lambda entry: -entry[0])
            rendered = mixed + rendered[cfg.rerank_candidates :]
    if time.monotonic() >= deadline:
        raise EmbeddingError("Search exceeded the service deadline.")
    overview = cfg.overview_outline and intent.wants_overview(question)
    if overview:
        rendered = _round_robin(rendered, index)
    selected, used = [], 0
    top_score = rendered[0][0] if rendered else 0.0
    for score, item, order in rendered[:limit]:
        if (
            selected
            and len(selected) >= cfg.min_blocks
            and score < cfg.min_relative_score * top_score
        ):
            break
        size = _tokens(item["content"])
        if selected and used + size > cfg.context_tokens:
            break
        selected.append((item, order))
        used += size
    if cfg.order == "chronological":
        selected.sort(key=lambda entry: (entry[0]["created_at"], entry[1]))
    elif cfg.current_state_order and intent.wants_current_state(question):
        selected.sort(key=lambda entry: entry[0]["created_at"], reverse=True)  # newest first
    data = [item for item, _ in selected]
    extras = []
    if overview:
        extras.append(outline_item(index))
    if cfg.profile_digest and intent.wants_recommendation(question):
        extras.append(profile_item(index, scores, cfg.profile_items))
    data = [e for e in extras if e] + data
    if cfg.redact_secrets:
        for item in data:
            item["content"] = intent.redact_secrets(item["content"], question)
    return data[: request.top_k]

"""The Search pipeline: query views -> fusion -> excerpts -> rerank -> selection -> final shape."""

import re
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
    structured_item,
)
from plain_memory.text import encoding


def _tokens(text: str) -> int:
    return len(encoding().encode(text, disallowed_special=()))


def query_with_options(question: str, options: list[str] | None) -> str:
    """Some callers already include the choices in query as well as options."""
    present = " ".join(question.split())
    lines = {" ".join(line.split()) for line in question.splitlines()}
    seen = set()
    missing = []
    for option in options or []:
        cleaned = " ".join(option.split())
        # A short, unlabelled answer can occur incidentally in the question.
        # Only elide whole choice lines or an explicitly labelled choice.
        labelled = re.match(r"^(?:[A-Z][.)]|\([A-Z]\))\s", cleaned)
        included = cleaned in lines or (
            labelled and re.search(r"(?<!\w)" + re.escape(cleaned) + r"(?!\w)", present)
        )
        if cleaned and not included and cleaned not in seen:
            missing.append(option)
        seen.add(cleaned)
    return question + ("\nOptions:\n" + "\n".join(missing) if missing else "")


def _truncate_item(item: dict, budget: int) -> dict:
    """Mark partial evidence explicitly and keep Unicode intact at token boundaries."""
    suffix = "\n[Excerpt truncated to fit context budget]"
    tokens = encoding().encode(item["content"], disallowed_special=())
    prefix = tokens[: max(0, budget - _tokens(suffix))]
    content = encoding().decode(prefix, errors="ignore") + suffix
    while _tokens(content) > budget and prefix:
        prefix = prefix[:-1]
        content = encoding().decode(prefix, errors="ignore") + suffix
    return {**item, "content": content}


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
    full_query = query_with_options(question, request.options)
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
        (
            score,
            render_block(index, key, lo, hi, cfg.show_notes, cfg.show_facts, cfg.structured_state),
            (key, lo),
        )
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
    extras = []
    if cfg.structured_state:
        extras.append(structured_item(index, question))
    if overview:
        extras.append(outline_item(index))
    if cfg.profile_digest and intent.wants_recommendation(question):
        extras.append(profile_item(index, scores, cfg.profile_items))

    def prepare(item):
        if cfg.redact_secrets:
            return {**item, "content": intent.redact_secrets(item["content"], question)}
        return item

    # Summaries consume both item slots and tokens. Reserve most of the token
    # budget for original evidence even when an outline covers many sessions.
    data, used = [], 0
    for extra in extras:
        if extra is None or len(data) >= limit:
            continue
        extra = prepare(extra)
        budget = cfg.context_tokens // 4 - used
        if budget <= _tokens("\n[Excerpt truncated to fit context budget]"):
            break
        if _tokens(extra["content"]) > budget:
            extra = _truncate_item(extra, budget)
        data.append(extra)
        used += _tokens(extra["content"])
    selected = []
    top_score = rendered[0][0] if rendered else 0.0
    for score, item, order in rendered:
        if len(data) + len(selected) >= limit:
            break
        if (
            selected
            and len(selected) >= cfg.min_blocks
            and score < cfg.min_relative_score * top_score
        ):
            continue  # session diversification can put a higher score later
        item = prepare(item)
        size = _tokens(item["content"])
        if used + size > cfg.context_tokens:
            if selected:
                continue  # a shorter later excerpt can still fit
            item = _truncate_item(item, cfg.context_tokens - used)
            size = _tokens(item["content"])
        selected.append((item, order))
        used += size
    if cfg.order == "chronological":
        selected.sort(key=lambda entry: (entry[0]["created_at"], entry[1]))
    elif cfg.sequence_order and intent.wants_sequence(question):
        selected.sort(key=lambda entry: (entry[0]["created_at"], entry[1]))  # oldest first
    elif cfg.current_state_order and intent.wants_current_state(question):
        selected.sort(key=lambda entry: entry[0]["created_at"], reverse=True)  # newest first
    return data + [item for item, _ in selected]

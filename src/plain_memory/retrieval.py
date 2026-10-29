"""Per-user hybrid retrieval: dense + BM25 (bm25s) -> RRF -> window blocks -> optional rerank."""

import hashlib
import json
import threading
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta, timezone

import bm25s
import numpy as np

from plain_memory import intent, timeparse
from plain_memory.config import RetrievalConfig
from plain_memory.embedding import EmbeddingError
from plain_memory.text import lexical_terms

FB_KEY = "\x00feedback"  # pseudo group key of single graded-attempt excerpts


def interleave(vector_ids: list[str], bm25_ids: list[str], top_k: int) -> list[str]:
    result, seen = [], set()
    for index in range(max(len(vector_ids), len(bm25_ids))):
        for candidates in (vector_ids, bm25_ids):
            if index < len(candidates) and candidates[index] not in seen:
                seen.add(candidates[index])
                result.append(candidates[index])
                if len(result) == top_k:
                    return result
    return result


def format_time(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    moment = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=timestamp_ms)
    text = f"{moment:%A}, {moment.day} {moment:%B %Y}"
    if moment.hour or moment.minute:
        hour = moment.hour % 12 or 12
        text += f", {hour}:{moment:%M} {'AM' if moment.hour < 12 else 'PM'}"
    return text


def _speaker_prefixed(row: dict) -> str:
    head = row["content"][:48]
    colon = head.find(":")
    if 0 < colon <= 40 and "\n" not in head[:colon]:
        return row["content"]
    return f"{row['role']}: {row['content']}"


class UserIndex:
    """Immutable snapshot of one user's memory with dense/BM25/sequence structures."""

    def __init__(self, rows: list[dict]):
        try:
            self.matrix = np.stack([np.frombuffer(row["embedding"], dtype="<f4") for row in rows])
            if not np.isfinite(self.matrix).all():
                raise ValueError
        except (ValueError, TypeError):
            raise EmbeddingError("Stored vectors are invalid.") from None
        self.rows = rows
        self.days = [
            timeparse.unit_day(row["source_timestamp_ms"]).toordinal()
            if row["source_timestamp_ms"] is not None and row["kind"] == "msg"
            else None
            for row in rows
        ]
        self.feedback = {i: row["note"] for i, row in enumerate(rows) if row["kind"] == "fb"}
        terms = [json.loads(row["terms_json"]) for row in rows]
        self.searchable = [i for i, t in enumerate(terms) if t]
        self.bm25 = None
        if self.searchable:
            self.bm25 = bm25s.BM25()
            self.bm25.index([terms[i] for i in self.searchable], show_progress=False)
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            if row["kind"] == "msg":
                groups[row["request_id"]].append(i)
        self.groups: dict[str, list[int]] = {}
        self.position: dict[int, tuple[str, int]] = {}
        for key, members in groups.items():
            members.sort(key=lambda i: (rows[i]["message_index"], rows[i]["chunk_index"]))
            self.groups[key] = members
            for pos, i in enumerate(members):
                self.position[i] = (key, pos)
        anchors = {
            (row["request_id"], row["message_index"]): i
            for i, row in enumerate(rows)
            if row["kind"] == "msg" and row["chunk_index"] == 0
        }
        self.anchor: dict[int, int] = {}
        self.facts: dict[int, list[str]] = defaultdict(list)
        for i, row in enumerate(rows):
            if row["kind"] == "fact":
                target = anchors.get((row["request_id"], row["message_index"]))
                if target is not None:
                    self.anchor[i] = target
                    self.facts[target].append(row["content"])

    def day_units(self, constraints) -> list[int]:
        spans = [(a.toordinal(), b.toordinal()) for a, b in constraints]
        return [
            i
            for i, day in enumerate(self.days)
            if day is not None and any(a <= day <= b for a, b in spans)
        ]

    def dense(self, vector: np.ndarray, limit: int) -> list[int]:
        if self.matrix.shape[1] != len(vector):
            raise EmbeddingError("Stored vectors are incompatible with the query model.")
        sims = self.matrix @ vector
        return list(np.argsort(-sims, kind="stable")[:limit])

    def lexical(self, query: str, limit: int) -> list[int]:
        tokens = lexical_terms(query)
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(-scores, kind="stable")[:limit]
        return [self.searchable[j] for j in order if scores[j] > 0]


class IndexCache:
    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.items: OrderedDict[str, tuple[tuple, UserIndex]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, store, user_id: str) -> UserIndex | None:
        fingerprint = store.fingerprint(user_id)
        if fingerprint[0] == 0:
            return None
        with self.lock:
            cached = self.items.get(user_id)
            if cached and cached[0] == fingerprint:
                self.items.move_to_end(user_id)
                return cached[1]
            rows = store.snapshot(user_id)
            index = UserIndex(rows)
            self.items[user_id] = (fingerprint, index)
            self.items.move_to_end(user_id)
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)
            return index


def fuse(
    index: UserIndex,
    queries: list[tuple[str, float]],
    vectors: np.ndarray,
    config: RetrievalConfig,
    original_query: str = "",
) -> dict[int, float]:
    """Weighted RRF over dense and lexical rankings of every query view, plus two nudges:
    messages written on a date the question names, and graded streaming attempts."""
    scores: dict[int, float] = defaultdict(float)
    limit = max(config.candidate_units * 3, 100)
    for (text, weight), vector in zip(queries, vectors):
        for ranking in (index.dense(vector, limit), index.lexical(text, limit)):
            for rank, unit in enumerate(ranking):
                scores[unit] += weight / (config.rrf_k + rank + 1)
    if config.date_boost > 0:
        constraints = timeparse.query_dates(original_query)
        if constraints:
            for unit in index.day_units(constraints):
                scores[unit] += config.date_boost
    for unit, outcome in index.feedback.items():
        if unit in scores:
            if outcome == "passed":
                scores[unit] *= config.feedback_pass_boost
            elif outcome == "failed":
                scores[unit] *= config.feedback_fail_penalty
    return scores


def build_blocks(index: UserIndex, unit_scores: dict[int, float], config: RetrievalConfig):
    """Expand top units to neighbouring messages and merge overlaps -> [(score, key, lo, hi)]."""
    best: dict[int, float] = {}
    attempts = []
    for unit, score in sorted(unit_scores.items(), key=lambda kv: -kv[1])[: config.candidate_units]:
        if unit in index.feedback:
            attempts.append((score, FB_KEY, unit, unit))
            continue
        anchor = index.anchor.get(unit, unit)
        if anchor in index.position and score > best.get(anchor, 0.0):
            best[anchor] = score
    per_group: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for anchor, score in best.items():
        key, pos = index.position[anchor]
        size = len(index.groups[key])
        lo, hi = max(0, pos - config.window_before), min(size - 1, pos + config.window_after)
        per_group[key].append((lo, hi, score, pos))
    blocks = []
    for key, spans in per_group.items():
        spans.sort()
        merged = []
        for lo, hi, score, pos in spans:
            if merged and lo <= merged[-1][1] + 1:
                last = merged[-1]
                last[1] = max(last[1], hi)
                last[2].append((score, pos))
            else:
                merged.append([lo, hi, [(score, pos)]])
        for lo, hi, hits in merged:
            start = lo
            while start <= hi:
                end = min(hi, start + config.max_block_units - 1)
                inside = [s for s, p in hits if start <= p <= end]
                blocks.append((max(inside, default=0.0), key, start, end))
                start = end + 1
    blocks.extend(attempts[:5])
    blocks.sort(key=lambda b: (-b[0], b[1], b[2]))
    return blocks


def render_block(
    index: UserIndex, key: str, lo: int, hi: int, show_notes: bool, show_facts: bool = False
) -> dict:
    rows = index.rows
    if key == FB_KEY:
        row = rows[lo]
        identity = [row["user_id"], key, row["id"]]
        return {
            "id": "mem_" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:32],
            "content": row["content"],
            "created_at": row["created_at"],
        }
    members = index.groups[key][lo : hi + 1]
    first_time = next(
        (
            rows[i]["source_timestamp_ms"]
            for i in members
            if rows[i]["source_timestamp_ms"] is not None
        ),
        None,
    )
    header = format_time(first_time)
    lines = [f"[Conversation excerpt{' | ' + header if header else ''}]"]
    seen_day = None
    if first_time is not None:
        seen_day = format_time(first_time).split(", ")[1]
    facts = []
    for i in members:
        row = rows[i]
        text = _speaker_prefixed(row)
        stamp = row["source_timestamp_ms"]
        if stamp is not None and format_time(stamp).split(", ")[1] != seen_day:
            text = f"({format_time(stamp)}) {text}"
        lines.append(text)
        if show_notes and row["note"]:
            lines.append(f"  (time note: {row['note']})")
        if show_facts:
            facts.extend(index.facts.get(i, []))
    if facts:
        lines.append("Facts: " + " | ".join(dict.fromkeys(facts)))
    identity = [rows[members[0]]["user_id"], key, rows[members[0]]["id"], rows[members[-1]]["id"]]
    return {
        "id": "mem_" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:32],
        "content": "\n".join(lines),
        "created_at": rows[members[0]]["created_at"],
    }


def _digest_id(user_id: str, kind: str, body: str) -> str:
    return "mem_" + hashlib.sha256(f"{user_id}\x00{kind}\x00{body}".encode()).hexdigest()[:32]


def profile_item(index: UserIndex, scores: dict[int, float], limit: int) -> dict | None:
    """Earlier statements the user made about themselves, picked by relevance to the question."""
    lines, seen = [], set()
    for unit, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:40]:
        row = index.rows[unit]
        if row["kind"] != "msg" or row["role"] != "user":
            continue
        for sentence in intent.first_person_sentences(row["content"], limit=2):
            if sentence.lower() in seen:
                continue
            seen.add(sentence.lower())
            when = timeparse.unit_day(row["source_timestamp_ms"])
            lines.append(f"- ({timeparse.fmt_day(when) if when else 'undated'}) {sentence}")
        if len(lines) >= limit:
            break
    if not lines:
        return None
    body = "\n".join(lines[:limit])
    first = index.rows[next(iter(scores))]
    return {
        "id": _digest_id(first["user_id"], "profile", body),
        "content": "[User profile notes: things the user said about themselves earlier, "
        "relevant for personalising the answer]\n" + body,
        "created_at": first["created_at"],
    }


def outline_item(index: UserIndex, max_sessions: int = 60) -> dict | None:
    """One line per session (date + how it opened) so overview questions see the whole history."""
    sessions: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(index.rows):
        if row["kind"] == "msg":
            sessions[row["session_id"]].append(i)
    if not sessions:
        return None
    ordered = sorted(
        sessions.values(),
        key=lambda members: (
            index.rows[members[0]]["source_timestamp_ms"] or 0,
            index.rows[members[0]]["id"],
        ),
    )
    if len(ordered) > max_sessions:  # even sample keeps both ends of the timeline
        step = (len(ordered) - 1) / (max_sessions - 1)
        ordered = [ordered[round(k * step)] for k in range(max_sessions)]
    lines = []
    for members in ordered:
        opener = next(
            (index.rows[i] for i in members if index.rows[i]["role"] == "user"),
            index.rows[members[0]],
        )
        when = timeparse.unit_day(opener["source_timestamp_ms"])
        text = " ".join(opener["content"].split())[:170]
        lines.append(
            f"- {timeparse.fmt_day(when) if when else 'undated'} ({len(members)} messages): {text}"
        )
    body = "\n".join(lines)
    first = index.rows[0]
    return {
        "id": _digest_id(first["user_id"], "outline", body),
        "content": "[Overview of the whole history, oldest session first; one line per session]\n"
        + body,
        "created_at": first["created_at"],
    }

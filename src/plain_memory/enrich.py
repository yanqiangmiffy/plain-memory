"""Add-time gpt-4o-mini enrichment: resolved time notes + atomic facts per dialogue group."""

import concurrent.futures

from plain_memory.llm import LLM
from plain_memory.text import encoding

SYSTEM = """You turn chat logs into searchable long-term memory. Input lines look like
[3] Caroline: I went to a support group yesterday.
and the header gives the date the lines were written.

Return JSON: {"facts": [{"i": <line number>, "text": "<fact>"}], "time_notes": [{"i": <line number>, "note": "<resolution>"}]}

facts: atomic, self-contained statements (one idea each) in third person using the speakers' real names, never pronouns. Cover stable facts, preferences, relationships, possessions, plans, events, activities, opinions, feelings, numbers, titles and proper names that a later question might ask about. Express time exactly as precisely as the source: if a line says "yesterday" give that date; if it says "last week", "two weekends ago", "last month" or "last year" write the same relative period anchored to the header date (e.g. "during the week before 9 June 2023", "in May 2023", "in 2022") and NEVER invent a specific day. Skip greetings and filler. "i" is the line that supports the fact.

time_notes: for every line containing a relative time expression (yesterday, last week, next month, two weekends ago, last year, this Friday, a few days ago, ...), give the resolution relative to the header date. Day-level expressions -> an explicit date ("yesterday = 7 May 2023"). Week-level -> keep the relative form and add the range ("last week = the week before 9 June 2023, i.e. 29 May - 4 June 2023"). Month/year-level -> the month or year ("last year = 2022"). Only include lines that really contain such an expression.

Output JSON only."""


def _groups(lines: list[str], limit: int) -> list[list[int]]:
    """Consecutive line indexes whose token total stays under the limit."""
    groups, current, used = [], [], 0
    for index, line in enumerate(lines):
        size = len(encoding().encode(line, disallowed_special=()))
        if current and used + size > limit:
            groups.append(current)
            current, used = [], 0
        current.append(index)
        used += size
    if current:
        groups.append(current)
    return groups


def enrich(
    lines: list[str], header: str, llm: LLM, deadline: float, group_tokens: int
) -> tuple[dict[int, str], list[tuple[int, str]]]:
    """Return ({line index: time note}, [(line index, fact)]). Failures yield empty parts."""
    notes: dict[int, str] = {}
    facts: list[tuple[int, str]] = []

    def run(group: list[int]):
        body = "\n".join(f"[{i}] {lines[i]}" for i in group)
        result = llm.json(SYSTEM, f"Date: {header}\n\n{body}", 3000, deadline)
        return group, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for group, result in pool.map(run, _groups(lines, group_tokens)):
            if not isinstance(result, dict):
                continue
            allowed = set(group)
            for item in result.get("time_notes") or []:
                try:
                    index, note = int(item["i"]), str(item["note"]).strip()
                except (KeyError, TypeError, ValueError):
                    continue
                if index in allowed and note:
                    notes[index] = note
            for item in result.get("facts") or []:
                try:
                    index, text = int(item["i"]), str(item["text"]).strip()
                except (KeyError, TypeError, ValueError):
                    continue
                if index in allowed and text:
                    facts.append((index, text))
    return notes, facts


QUERY_SYSTEM = """You help a memory search engine. Given a question (and optional answer options)
about a long chat history, return JSON {"queries": [...]} with 2-3 short search queries that
together cover the evidence needed: name every person/entity, split multi-part or multi-hop
questions into their parts, and for "when"/"how long"/"how many" questions add a query for the
underlying event itself. Never answer the question. Keep each query under 20 words. Output JSON only."""


def expand_query(query: str, options: list[str] | None, llm: LLM, deadline: float) -> list[str]:
    body = query + ("\nOptions:\n" + "\n".join(options) if options else "")
    result = llm.json(QUERY_SYSTEM, body, 300, deadline)
    if not isinstance(result, dict):
        return []
    queries = result.get("queries")
    if not isinstance(queries, list):
        return []
    return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:3]

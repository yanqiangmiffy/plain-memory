"""Source-backed views of structured facts and explicit replacements.

No entity names, task IDs, answers or world knowledge live here. Ordinary
conflicting assertions remain unresolved; only an explicit replace operation
supersedes an earlier fact. Original memory is always preserved.
"""

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass

from plain_memory import timeparse


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    subject_key: str
    relation_key: str
    object_key: str
    replace: bool


def parse_fact(text: str) -> Fact | None:
    if len(text) > 16000:
        return None
    start = text.find("{")
    if start < 0:
        return None
    prefix = text[:start].strip()
    # Only a data envelope or a bare object; never interpret JSON embedded in
    # quoted instructions, prose, code examples or graded answer attempts.
    update = bool(
        re.fullmatch(
            r"(?:Corpus:\s*)?UPDATE:\s*replace the prior value of this subject and relation "
            r"with the following fact\.",
            prefix,
            re.I,
        )
    )
    if prefix not in ("", "Corpus:") and not update:
        return None
    try:
        value = json.loads(text[start:])
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None
    labels = [value.get(k) for k in ("subject", "relation", "object")]
    if any(not isinstance(v, str) or not v.strip() or len(v) > 500 for v in labels):
        return None
    operation = value.get("operation")
    if operation is not None and operation not in ("assert", "replace"):
        return None
    keys = []
    for field, label in zip(("subject_id", "relation_id", "object_id"), labels):
        identity = value.get(field)
        if identity is not None and (not isinstance(identity, str) or not identity.strip()):
            return None
        keys.append("id:" + identity if identity else "label:" + _norm(label))
    return Fact(*(v.strip() for v in labels), *keys, update or operation == "replace")


class StructuredState:
    def __init__(self, rows: list[dict]):
        self.facts = {}
        self.superseded: dict[int, int] = {}
        self.active: dict[tuple[str, str], list[int]] = defaultdict(list)
        for unit, row in enumerate(rows):
            if row["kind"] != "msg" or row["role"] != "user" or row["chunk_index"] != 0:
                continue
            fact = parse_fact(row["content"])
            if fact is None:
                continue
            self.facts[unit] = fact
            key = fact.subject_key, fact.relation_key
            if fact.replace:
                for previous in self.active[key]:
                    self.superseded[previous] = unit
                self.active[key] = []
            self.active[key].append(unit)
        self.outgoing: dict[str, list[int]] = defaultdict(list)
        self.names: dict[str, set[str]] = defaultdict(set)
        for members in self.active.values():
            for unit in members:
                fact = self.facts[unit]
                self.outgoing[fact.subject_key].append(unit)
                self.names[fact.subject_key].add(fact.subject)
                self.names[fact.object_key].add(fact.object)

    def connected(self, query: str, limit: int = 32, max_hops: int = 6) -> list[int]:
        """Follow outgoing evidence links from entities actually named in the query."""
        if (
            not self.superseded
            or timeparse.explicit_dates(query)
            or re.search(
                r"\b(before|previously|formerly|historical|used to|was|were|did|had|"
                r"yesterday|last (?:week|month|year)|at that time)\b|"
                r"(?<!\w)(?:19|20)\d{2}(?!\w)|以前|之前|历史|去年|上个月|当时",
                query,
                re.I,
            )
        ):
            return []  # no undated 'current' view for historical questions
        seeds = [
            key
            for key, names in self.names.items()
            if any(
                re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", query, re.I) for name in names
            )
        ]
        queue = deque((key, 0) for key in seeds)
        visited, picked, seen_units = set(), [], set()
        while queue and len(picked) < limit:
            key, depth = queue.popleft()
            if key in visited or depth >= max_hops:
                continue
            visited.add(key)
            for unit in self.outgoing.get(key, []):
                if unit not in seen_units:
                    picked.append(unit)
                    seen_units.add(unit)
                    queue.append((self.facts[unit].object_key, depth + 1))
                    if len(picked) == limit:
                        break
        if not any(self.facts[unit].replace for unit in picked):
            return []
        return picked

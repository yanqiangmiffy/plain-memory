"""Cheap, deterministic question-intent detection and returned-text safety helpers."""

import re
from datetime import datetime

_CURRENT = re.compile(
    r"\b(?:currently|right now|nowadays|these days|at the moment|presently|latest|most recent(?:ly)?|"
    r"up to date|these days|no longer|anymore|any more|still)\b|"
    r"\b(?:what|which|where|who|how)\b[^?.]{0,40}\b(?:is|are|am)\b[^?.]{0,60}\bnow\b",
    re.I,
)
_RECOMMEND = re.compile(
    r"\b(?:recommend\w*|suggest\w*|any ideas|ideas for|tips?\b|advice|what should i|should i|"
    r"can you help me (?:find|choose|pick|plan|decide)|help me (?:find|choose|pick|plan|decide)|"
    r"what (?:can|could) i (?:do|try|use|get|make|cook|watch|read|play|buy)|"
    r"looking for (?:a|an|some)|i(?:'m| am) (?:thinking|planning|looking) (?:of|to|for)|"
    r"how (?:can|do|should) i (?:improve|get better|learn))",
    re.I,
)
_GLOBAL = re.compile(
    r"\b(?:summari[sz]e|summary of|overview of|recap|in summary|"
    r"everything (?:we|i)(?:'ve| have)? (?:discussed|talked|covered|said)|"
    r"what (?:have|did) we (?:discussed?|talk(?:ed)? about|cover(?:ed)?)|"
    r"across (?:all )?(?:our |my |the )?(?:sessions|conversations|chats)|"
    r"(?:entire|whole|full|complete) (?:conversation |chat )?history|overall (?:picture|story|journey|progress))\b",
    re.I,
)


def wants_current_state(query: str) -> bool:
    return bool(_CURRENT.search(query))


def wants_recommendation(query: str) -> bool:
    return bool(_RECOMMEND.search(query))


def wants_overview(query: str) -> bool:
    return bool(_GLOBAL.search(query))


_FIRST_PERSON = re.compile(r"\b(?:I|I'm|I've|I'd|I'll|my|me|mine|myself|we|our|we're|we've)\b")
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


def first_person_sentences(text: str, limit: int = 3, width: int = 220) -> list[str]:
    """User sentences that describe the user (their things, habits, plans)."""
    picked = []
    for sentence in _SENTENCES.split(text):
        sentence = sentence.strip()
        if 12 <= len(sentence) and _FIRST_PERSON.search(sentence) and not sentence.endswith("?"):
            picked.append(sentence[:width])
            if len(picked) == limit:
                break
    return picked


# ---- secrets ----------------------------------------------------------------------------

_SECRET_ASK = re.compile(
    r"pass(?:word|code|phrase)|\bpwd\b|secret|token|api[ _-]?key|\bssn\b|social security|"
    r"credit card|card number|account number|\bpin\b|private key",
    re.I,
)
_SECRETS = [
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "[redacted key]"),
    (re.compile(r"(?i)\b(pass(?:word|code|phrase)|pwd|secret|api[ _-]?key|access[ _-]?token|token)\b(\s*(?:is|was|=|:)\s*)(?!\[redacted)\S{4,}"), r"\1\2[redacted]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[redacted id number]"),
    (re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b"), "[redacted card number]"),
]  # fmt: skip


def redact_secrets(text: str, query: str) -> str:
    """Mask obvious credentials in text unless the question itself asks for them."""
    if _SECRET_ASK.search(query):
        return text
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    return text


def short_date(moment: datetime | None) -> str:
    return f"{moment.day} {moment:%B %Y}" if moment else "undated"

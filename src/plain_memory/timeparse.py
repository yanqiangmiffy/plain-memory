"""Deterministic time helpers (no LLM): relative-time notes for Add, date constraints for Search.

`relative_notes(text, moment)` resolves phrases such as "yesterday" or "last week" against the
moment a message was written and returns short notes like "yesterday = 7 May 2023". Granularity
is kept as coarse as the phrase: days stay days, weeks stay weeks, months stay months.

`query_dates(text)` finds explicit day/month dates in a question; `unit_range(...)` gives the
day a stored message belongs to, so Search can favour messages written on a queried date.
"""

import calendar
import re
from datetime import date, datetime, timedelta, timezone

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
MONTHS["sept"] = 9
_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}  # fmt: skip
WEEKDAYS = [d.lower() for d in calendar.day_name]
_COUNT = r"(\d{1,3}|" + "|".join(NUMBER_WORDS) + r")"


def moment_of(timestamp_ms: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=timestamp_ms)


def fmt_day(d: date) -> str:
    return f"{d.day} {calendar.month_name[d.month]} {d.year}"


def fmt_range(a: date, b: date) -> str:
    if a.year == b.year:
        left = f"{a.day} {calendar.month_name[a.month]}"
        return f"{left} - {fmt_day(b)}"
    return f"{fmt_day(a)} - {fmt_day(b)}"


def _count(token: str) -> int:
    token = token.lower()
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def _shift_month(d: date, months: int) -> tuple[int, int]:
    index = d.year * 12 + (d.month - 1) + months
    return index // 12, index % 12 + 1


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


_RULES = [
    ("yesterday", re.compile(r"\byesterday\b", re.I)),
    ("last night", re.compile(r"\blast night\b", re.I)),
    ("tomorrow", re.compile(r"\btomorrow\b", re.I)),
    ("days ago", re.compile(rf"\b{_COUNT} days? ago\b", re.I)),
    ("weeks ago", re.compile(rf"\b{_COUNT} weeks? ago\b", re.I)),
    ("months ago", re.compile(rf"\b{_COUNT} months? ago\b", re.I)),
    ("years ago", re.compile(rf"\b{_COUNT} years? ago\b", re.I)),
    ("last week", re.compile(r"\blast week\b", re.I)),
    ("next week", re.compile(r"\bnext week\b", re.I)),
    ("last month", re.compile(r"\blast month\b", re.I)),
    ("next month", re.compile(r"\bnext month\b", re.I)),
    ("last year", re.compile(r"\blast year\b", re.I)),
    ("next year", re.compile(r"\bnext year\b", re.I)),
    ("weekends ago", re.compile(rf"\b{_COUNT} weekends? ago\b", re.I)),
    ("last weekend", re.compile(r"\blast weekend\b", re.I)),
    ("weekday", re.compile(rf"\b(last|next) ({'|'.join(WEEKDAYS)})\b", re.I)),
]


def relative_notes(text: str, moment: datetime) -> list[str]:
    """Notes for every recognised relative expression in `text`, anchored at `moment`."""
    today = moment.date()
    notes: list[str] = []
    for name, pattern in _RULES:
        for match in pattern.finditer(text):
            phrase = match.group(0).lower()
            note = None
            if name in ("yesterday", "last night"):
                note = fmt_day(today - timedelta(days=1))
            elif name == "tomorrow":
                note = fmt_day(today + timedelta(days=1))
            elif name == "days ago":
                note = fmt_day(today - timedelta(days=_count(match.group(1))))
            elif name in ("weeks ago", "last week"):
                n = _count(match.group(1)) if name == "weeks ago" else 1
                start = _monday(today) - timedelta(weeks=n)
                note = (
                    f"the week before {fmt_day(today)} ({fmt_range(start, start + timedelta(days=6))})"
                    if n == 1
                    else f"{n} weeks before {fmt_day(today)} (week of {fmt_range(start, start + timedelta(days=6))})"
                )
            elif name == "next week":
                start = _monday(today) + timedelta(weeks=1)
                note = f"the week after {fmt_day(today)} ({fmt_range(start, start + timedelta(days=6))})"
            elif name in ("months ago", "last month"):
                n = _count(match.group(1)) if name == "months ago" else 1
                year, month = _shift_month(today, -n)
                note = f"{calendar.month_name[month]} {year}"
            elif name == "next month":
                year, month = _shift_month(today, 1)
                note = f"{calendar.month_name[month]} {year}"
            elif name in ("years ago", "last year"):
                n = _count(match.group(1)) if name == "years ago" else 1
                note = str(today.year - n)
            elif name == "next year":
                note = str(today.year + 1)
            elif name in ("weekends ago", "last weekend"):
                n = _count(match.group(1)) if name == "weekends ago" else 1
                saturday = today - timedelta(days=(today.weekday() - 5) % 7 or 7)
                saturday -= timedelta(weeks=n - 1)
                note = f"the weekend of {fmt_range(saturday, saturday + timedelta(days=1))}"
            elif name == "weekday":
                direction, day = match.group(1).lower(), WEEKDAYS.index(match.group(2).lower())
                delta = (today.weekday() - day) % 7 or 7
                target = (
                    today - timedelta(days=delta)
                    if direction == "last"
                    else (today + timedelta(days=(day - today.weekday()) % 7 or 7))
                )
                note = fmt_day(target)
            if note:
                notes.append(f"{phrase} = {note}")
    return list(dict.fromkeys(notes))


# ---- explicit dates in questions / text ---------------------------------------------------

_DAY_FIRST = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})\.?,?\s+(\d{{4}})\b", re.I)
_MONTH_FIRST = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_MONTH_YEAR = re.compile(rf"\b({_MONTH})\.?,?\s+(\d{{4}})\b", re.I)
_ISO = re.compile(r"(?<![\d-])(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?(?![\d-])")


def _safe(y: int, m: int, d: int | None) -> tuple[date, date] | None:
    try:
        if d is not None:
            return date(y, m, d), date(y, m, d)
        return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
    except ValueError:
        return None


def explicit_dates(text: str) -> list[tuple[date, date]]:
    """Day- or month-level dates written in `text` as inclusive (start, end) ranges."""
    found: list[tuple[date, date]] = []
    consumed: list[tuple[int, int]] = []

    def free(span):
        return all(span[1] <= a or span[0] >= b for a, b in consumed)

    for pattern, build in (
        (_DAY_FIRST, lambda g: (int(g[2]), MONTHS[g[1].lower()], int(g[0]))),
        (_MONTH_FIRST, lambda g: (int(g[2]), MONTHS[g[0].lower()], int(g[1]))),
        (_ISO, lambda g: (int(g[0]), int(g[1]), int(g[2]) if g[2] else None)),
        (_MONTH_YEAR, lambda g: (int(g[1]), MONTHS[g[0].lower()], None)),
    ):
        for match in pattern.finditer(text):
            if not free(match.span()):
                continue
            span = _safe(*build(match.groups()))
            if span:
                found.append(span)
                consumed.append(match.span())
    return found


def query_dates(query: str) -> list[tuple[date, date]]:
    """Date constraints of a question. Ranges with "before/after/between" are ignored."""
    if re.search(r"\b(before|after|between|since|until|from)\b", query, re.I):
        return []
    return explicit_dates(query)[:2]


def unit_day(timestamp_ms: int | None) -> date | None:
    return moment_of(timestamp_ms).date() if timestamp_ms is not None else None


def day_matches(day: date | None, constraints: list[tuple[date, date]]) -> bool:
    return day is not None and any(a <= day <= b for a, b in constraints)

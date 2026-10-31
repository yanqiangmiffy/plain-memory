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

MONTHS = {"sept": 9}
for _number in range(1, 13):
    MONTHS[calendar.month_name[_number].lower()] = _number
    MONTHS[calendar.month_abbr[_number].lower()] = _number
_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}  # fmt: skip
WEEKDAYS = [d.lower() for d in calendar.day_name]
_COUNT = r"(\d{1,3}|" + "|".join(NUMBER_WORDS) + r")"


def moment_of(timestamp_ms: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=timestamp_ms)


def message_timestamp(text: str) -> int | None:
    """Read a leading timestamp envelope, never dates mentioned inside a message.

    Offset-free envelope times use the service's UTC convention. The original
    text is retained so consumers can see exactly which time was supplied.
    """
    match = re.match(r"^\s*\[Time:\s*([^\]\n]{1,80})\]", text, re.I)
    if not match:
        return None
    value = match.group(1).strip()
    moment = None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        for form in ("%b %d, %Y, %H:%M:%S", "%B %d, %Y, %H:%M:%S"):
            try:
                moment = datetime.strptime(value, form)
                break
            except ValueError:
                continue
    if moment is None:
        return None
    moment = moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (moment - epoch) // timedelta(milliseconds=1)


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
    ("yesterday", re.compile(r"(?<!day before )\byesterday\b", re.I)),
    ("last night", re.compile(r"\blast night\b", re.I)),
    ("tomorrow", re.compile(r"(?<!day after )\btomorrow\b", re.I)),
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


# ---- wider coverage: seasons, named months, "in N units", "this ..." --------------------------

_UNIT = r"(days?|weeks?|months?|years?)"
_FUTURE_CUE = re.compile(
    r"\b(will|going to|gonna|plan(?:s|ning|ned)?|scheduled|due|leav(?:e|es|ing)|flight|fly(?:ing)?|"
    r"start(?:s|ing)?|begin(?:s|ning)?|expect(?:s|ed|ing)?|coming|arriv(?:e|es|ing)|"
    r"meeting|appointment|deadline|trip|opens?|ends?|exam|visit(?:ing)?|launch|release|"
    r"returns?)\b|'ll\b",
    re.I,
)
# Northern-hemisphere meteorological seasons: (first month, length in months).
_SEASONS = {"spring": (3, 3), "summer": (6, 3), "autumn": (9, 3), "fall": (9, 3), "winter": (12, 3)}
_MONTH_NAMES = "|".join(calendar.month_name[n] for n in range(1, 13))


def _plus_units(today: date, count: int, unit: str) -> str:
    unit = unit.rstrip("s")
    if unit == "day":
        return fmt_day(today + timedelta(days=count))
    if unit == "week":
        return "about " + fmt_day(today + timedelta(weeks=count))
    if unit == "month":
        year, month = _shift_month(today, count)
        return f"{calendar.month_name[month]} {year}"
    return str(today.year + count)


def _season_span(name: str, year: int) -> tuple[date, date]:
    """The season whose first day falls in `year` (winter runs into the next year)."""
    first, length = _SEASONS[name]
    start = date(year, first, 1)
    next_year, next_month = _shift_month(start, length)
    return start, date(next_year, next_month, 1) - timedelta(days=1)


def _season_note(match: re.Match, today: date) -> str:
    direction, name = match.group(1).lower(), match.group(2).lower()
    spans = [_season_span(name, year) for year in range(today.year - 2, today.year + 2)]
    if direction == "last":
        start, end = max((s for s in spans if s[1] < today), key=lambda s: s[0])
    elif direction == "next":
        start, end = min((s for s in spans if s[0] > today), key=lambda s: s[0])
    else:
        start, end = min((s for s in spans if s[1] >= today), key=lambda s: s[0])
    label = "autumn" if name == "fall" else name
    years = str(end.year) if start.year == end.year else f"{start.year}-{end.year}"
    return (
        f"{label} {years} ({calendar.month_name[start.month]} - "
        f"{calendar.month_name[end.month]} {end.year})"
    )


def _month_note(match: re.Match, today: date) -> str:
    direction, month = match.group(1).lower(), MONTHS[match.group(2).lower()]
    if direction == "last":
        year = today.year if month < today.month else today.year - 1
    elif direction == "next":
        year = today.year if month > today.month else today.year + 1
    else:
        year = today.year
    return f"{calendar.month_name[month]} {year}"


def _ahead_note(match: re.Match, today: date) -> str | None:
    """'in two weeks' only reads as a date when the sentence talks about the future."""
    around = match.string[max(0, match.start() - 80) : match.end() + 80]
    if match.group(0).lower().startswith("in ") and not _FUTURE_CUE.search(around):
        return None  # "finished it in two weeks" is a duration
    return _plus_units(today, _count(match.group(1)), match.group(2))


def _week_range(start: date) -> str:
    return fmt_range(start, start + timedelta(days=6))


def _this_note(match: re.Match, today: date) -> str:
    unit = match.group(1).lower()
    if unit == "week":
        return "the week of " + _week_range(_monday(today))
    if unit == "month":
        return f"{calendar.month_name[today.month]} {today.year}"
    return str(today.year)


def _weekend_note(match: re.Match, today: date) -> str:
    saturday = _monday(today) + timedelta(days=5)
    if match.group(1).lower() == "next":
        saturday += timedelta(weeks=1)
    return "the weekend of " + fmt_range(saturday, saturday + timedelta(days=1))


_EXTRA_RULES = [
    (
        re.compile(rf"\b(?i:(last|next|this)) ({'|'.join(_SEASONS)})\b", re.I),
        _season_note,
    ),
    (
        re.compile(rf"\b(?i:(last|next|this)) ({_MONTH_NAMES})\b"),
        _month_note,
    ),
    (re.compile(rf"\bin {_COUNT} {_UNIT}\b", re.I), _ahead_note),
    (re.compile(rf"\b{_COUNT} {_UNIT} from (?:now|today)\b", re.I), _ahead_note),
    (re.compile(r"\bthis (week|month|year)\b", re.I), _this_note),
    (re.compile(r"\b(this|next) weekend\b", re.I), _weekend_note),
    (
        re.compile(rf"\bthis ({'|'.join(WEEKDAYS)})\b", re.I),
        lambda m, today: fmt_day(
            _monday(today) + timedelta(days=WEEKDAYS.index(m.group(1).lower()))
        ),
    ),
    (
        re.compile(
            r"\b(a few|a couple of|a couple|several|a handful of) (days|weeks|months|years) ago\b",
            re.I,
        ),
        lambda m, today: f"{m.group(1).lower()} {m.group(2).lower()} before {fmt_day(today)}",
    ),
    (
        re.compile(r"\bthe day before yesterday\b", re.I),
        lambda m, t: fmt_day(t - timedelta(days=2)),
    ),
    (re.compile(r"\bthe day after tomorrow\b", re.I), lambda m, t: fmt_day(t + timedelta(days=2))),
    (re.compile(r"\b(today|tonight)\b", re.I), lambda m, today: fmt_day(today)),
]


def relative_notes(text: str, moment: datetime) -> list[str]:
    """Notes for every recognised relative expression in `text`, anchored at `moment`."""
    today = moment.date()
    notes: list[str] = []
    for pattern, resolve in _EXTRA_RULES:
        for match in pattern.finditer(text):
            note = resolve(match, today)
            if note:
                notes.append(f"{match.group(0).lower()} = {note}")
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

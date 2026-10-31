from datetime import datetime, timezone

import pytest

from plain_memory import intent, timeparse

MONDAY = datetime(2023, 5, 8, 14, 0, tzinfo=timezone.utc)


def notes(text, moment=MONDAY):
    return timeparse.relative_notes(text, moment)


def test_day_before_yesterday_and_day_after_tomorrow_are_not_read_as_yesterday_or_tomorrow():
    assert notes("the day before yesterday") == ["the day before yesterday = 6 May 2023"]
    assert notes("the day after tomorrow") == ["the day after tomorrow = 10 May 2023"]
    assert notes("yesterday") == ["yesterday = 7 May 2023"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("last summer", "last summer = summer 2022 (June - August 2022)"),
        ("this summer", "this summer = summer 2023 (June - August 2023)"),
        ("next winter", "next winter = winter 2023-2024 (December - February 2024)"),
        ("last fall", "last fall = autumn 2022 (September - November 2022)"),
    ],
)
def test_seasons_resolve_against_the_message_date(text, expected):
    assert notes(text) == [expected]


def test_winter_spans_the_new_year_boundary():
    january = datetime(2023, 1, 15, tzinfo=timezone.utc)
    assert notes("this winter", january) == [
        "this winter = winter 2022-2023 (December - February 2023)"
    ]
    assert notes("last winter", january) == [
        "last winter = winter 2021-2022 (December - February 2022)"
    ]


def test_named_months_pick_the_year_from_the_message_date():
    assert notes("last June") == ["last june = June 2022"]
    assert notes("next March") == ["next march = March 2024"]
    assert notes("last December", datetime(2023, 1, 2, tzinfo=timezone.utc)) == [
        "last december = December 2022"
    ]


def test_may_the_modal_verb_is_not_a_month():
    assert notes("this may help and it may be late") == []
    assert notes("this May is busy") == ["this may = May 2023"]


def test_future_offsets_need_a_future_cue_but_from_now_does_not():
    assert notes("I'm flying out in two weeks") == ["in two weeks = about 22 May 2023"]
    assert notes("the exam is in 3 days") == ["in 3 days = 11 May 2023"]
    assert notes("She finished the book in two weeks.") == []
    assert notes("10 days from now is the deadline") == ["10 days from now = 18 May 2023"]


def test_this_week_month_year_weekend_and_weekday():
    assert notes("this week") == ["this week = the week of 8 May - 14 May 2023"]
    assert notes("this month") == ["this month = May 2023"]
    assert notes("this year") == ["this year = 2023"]
    assert notes("this weekend") == ["this weekend = the weekend of 13 May - 14 May 2023"]
    assert notes("next weekend") == ["next weekend = the weekend of 20 May - 21 May 2023"]
    assert notes("this Friday") == ["this friday = 12 May 2023"]


def test_vague_offsets_stay_vague():
    assert notes("a few days ago") == ["a few days ago = a few days before 8 May 2023"]
    assert notes("several months ago") == ["several months ago = several months before 8 May 2023"]


def test_today_and_tonight_name_the_message_date():
    assert notes("Today is fine, tonight too") == [
        "today = 8 May 2023",
        "tonight = 8 May 2023",
    ]


def test_original_rules_still_apply_next_to_the_new_ones():
    friday = datetime(2023, 6, 9, tzinfo=timezone.utc)
    result = notes("yesterday and last week and last year", friday)
    assert result[0] == "yesterday = 8 June 2023"
    assert any(n.startswith("last week = ") for n in result)
    assert "last year = 2022" in result


@pytest.mark.parametrize(
    "question",
    [
        "In what order did I bring up the budget, the design and the launch?",
        "Which happened first, the move or the new job?",
        "Which came earlier: the trip to Rome or the wedding?",
        "Did the layoffs happen before or after the reorg?",
        "Describe the timeline of my project",
        "What is the chronological order of the releases?",
    ],
)
def test_sequence_questions_are_recognised(question):
    assert intent.wants_sequence(question)


@pytest.mark.parametrize(
    "question",
    [
        "What did she do last summer?",
        "Where does Mia live now?",
        "What did he buy after the party?",
    ],
)
def test_ordinary_questions_are_not_sequence_questions(question):
    assert not intent.wants_sequence(question)

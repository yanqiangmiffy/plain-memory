"""Streaming online-QA feedback: graded earlier answer attempts that arrive as ordinary messages.

The streaming track writes earlier question/answer pairs with a score into memory as user
messages shaped like

    Streaming online QA feedback:
    {"qa_id": "...", "question": "...", "predicted_answer": "...", "score": 1}

Such a message is a graded attempt, not a fact the user stated. A correct attempt (score 1) is
useful evidence for a later, similar question; a failed attempt (score 0) must not be repeated
and is demoted. Anything that does not match the envelope exactly stays an ordinary message.
"""

import json
import math
from dataclasses import dataclass

PREFIX = "Streaming online QA feedback:"
MAX_CHARS = 250_000


@dataclass(frozen=True)
class Attempt:
    question: str
    answer: str
    score: float

    @property
    def outcome(self) -> str:
        return "failed" if self.score <= 0 else "passed" if self.score >= 1 else "partial"

    def render(self) -> str:
        if self.outcome == "passed":
            head = "[Earlier attempt at this question was graded CORRECT (score 1)]"
        elif self.outcome == "failed":
            head = "[Earlier attempt at this question was graded INCORRECT (score 0); do not repeat it]"
        else:
            head = f"[Earlier attempt at this question was graded partially correct (score {self.score:g})]"
        return f"{head}\nQuestion: {self.question}\nAttempted answer: {self.answer}"


def parse(content: str, role: str) -> Attempt | None:
    if role != "user" or not content.startswith(PREFIX) or len(content) > MAX_CHARS:
        return None
    try:
        value = json.loads(content[len(PREFIX) :].strip())
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None
    question, answer, score = (
        value.get("question"),
        value.get("predicted_answer"),
        value.get("score"),
    )
    if not (isinstance(question, str) and question.strip() and isinstance(answer, str)):
        return None
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        return None
    return Attempt(question.strip(), answer.strip(), float(score))

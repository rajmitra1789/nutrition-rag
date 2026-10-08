"""Code-enforced refusals. These run before retrieval and generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

RefusalKind = Literal["out_of_scope", "not_in_corpus"]


@dataclass
class Refusal:
    kind: RefusalKind
    message: str


# Personal medical-advice requests. Asking what a document says about
# diet-related disease risk is allowed; asking the assistant to diagnose,
# treat, or prescribe is not.
_MEDICAL_PATTERNS = [
    re.compile(r"\b(diagnos(?:e|is|ing)|what(?:'s| is) wrong with me)\b", re.I),
    re.compile(r"\b(prescri(?:be|ption)|what medication|which (?:drug|medicine)|dosage|how many mg)\b", re.I),
    re.compile(r"\b(treat(?:ment|ing)? my|should i take|do i have|my symptoms?)\b", re.I),
    re.compile(
        r"\b(manage my (?:diabetes|hypertension|cancer|depression|anxiety)|insulin dose|blood pressure medication)\b",
        re.I,
    ),
    re.compile(r"\b(is this (?:a )?(?:heart attack|stroke|cancer|infection))\b", re.I),
    re.compile(r"\b(cure my|heal my|therapy for my)\b", re.I),
]

# Personal calorie or weight targets. Asking what a document states about
# energy percentages or a reference plate is allowed.
_WEIGHT_CALORIE_PATTERNS = [
    re.compile(
        r"\bhow many calories should (?:i|he|she|they|we|my \w+) (?:eat|have|consume|get|take)\b",
        re.I,
    ),
    re.compile(r"\b(?:my |a )?(?:daily |personal )?(?:calorie|kcal) (?:deficit|target|goal|budget|allowance|limit)\b", re.I),
    re.compile(r"\b(?:lose|gain|drop|shed)\s+\d+\s*(?:kg|kgs|lb|lbs|pounds|kilos|kilograms)\b", re.I),
    re.compile(r"\bwhat should (?:i|he|she|they|my (?:child|son|daughter|husband|wife|partner)) weigh\b", re.I),
    re.compile(r"\b(?:ideal|target|healthy) (?:weight|bmi) (?:for|of|should)\b", re.I),
    re.compile(r"\bbmi (?:goal|target|should)\b", re.I),
    re.compile(r"\b(?:give me|recommend|create|make) (?:a )?(?:weight[- ]loss|calorie) (?:diet|plan|target)\b", re.I),
    re.compile(r"\bhow much should (?:\w+ ){0,3}weigh\b", re.I),
    re.compile(r"\bwhat(?:'s| is) a (?:good|healthy) weight for (?:me|him|her|them)\b", re.I),
]

OUT_OF_SCOPE_MESSAGE = (
    "I cannot help with medical advice, personal calorie targets, or what anyone "
    "should weigh. Those questions need a qualified professional such as a "
    "registered dietitian, your GP, or another licensed clinician. I only answer "
    "from official public dietary guidance documents."
)


def classify_scope(query: str) -> Refusal | None:
    """Return an out-of-scope refusal when the query asks for forbidden help."""
    text = query.strip()
    if not text:
        return None
    for pattern in _MEDICAL_PATTERNS + _WEIGHT_CALORIE_PATTERNS:
        if pattern.search(text):
            return Refusal(kind="out_of_scope", message=OUT_OF_SCOPE_MESSAGE)
    return None


def format_searched_names(documents: list[dict[str, Any]]) -> str:
    if not documents:
        return "the loaded dietary guidance corpus"
    if len(documents) == 1:
        doc = documents[0]
        return f"{doc['document_name']} ({doc['publisher']}, {doc['year']})"
    lines = [f"- {doc['document_name']} ({doc['publisher']}, {doc['year']})" for doc in documents]
    return "the following documents:\n" + "\n".join(lines)


def not_in_corpus_refusal(documents: list[dict[str, Any]]) -> Refusal:
    searched = format_searched_names(documents)
    if len(documents) == 1:
        message = (
            f"The loaded guidance does not cover this question. I searched {searched} "
            "and did not find a passage that answers it."
        )
    else:
        message = (
            "The loaded guidance does not cover this question. I searched "
            f"{searched}\n\nNone of these documents contain a passage that answers it."
        )
    return Refusal(kind="not_in_corpus", message=message)

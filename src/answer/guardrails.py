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

# Personal health conditions and personal medical guidance. Condition words
# alone are allowed ("what does WHO say about cholesterol"); they refuse only
# with first-person framing or treatment intent. "kidney beans" is food.
_CONDITION = (
    r"(?:diabet(?:es|ic)|pregnan(?:t|cy)|breastfeeding|hypertension|"
    r"high blood pressure|(?:high )?cholesterol|kidney(?!\s+beans?\b)(?: disease)?|"
    r"heart disease|cancer|thyroid|pcos|ibs|c(?:o)?eliac|allerg(?:y|ies|ic)|"
    r"lactose intoleran(?:t|ce))"
)
_FIRST_PERSON = (
    r"(?:i am|i'm|im|i have|i've|i've got|i was|i got|i suffer from|"
    r"i'm being treated for|my (?:\w+ ){0,2})"
)

_PERSONAL_HEALTH_PATTERNS = [
    # "I have diabetes", "I am pregnant", "my high cholesterol", "my wife is pregnant"
    re.compile(rf"\b{_FIRST_PERSON}[^.?!]{{0,40}}?\b{_CONDITION}\b", re.I),
    # "should I eat ..." anywhere alongside a condition word
    re.compile(rf"^(?=.*\b(?:should|can|could|may) i (?:eat|drink|have)\b)(?=.*\b{_CONDITION}\b)", re.I | re.S),
    # "I am allergic", "my allergy", "I'm lactose intolerant"
    re.compile(r"\b(?:i am|i'm|im) (?:\w+ )?(?:allergic|lactose intolerant)\b", re.I),
    re.compile(r"\bmy (?:\w+ )?(?:allergy|allergies|intolerance)\b", re.I),
    # Treatment words that are personal on their own
    re.compile(r"\b(?:medications?|medicines?|my doctor|my gp|prescribed|diagnosed)\b", re.I),
    re.compile(r"\b(?:my|i have|i've got) (?:\w+ )?symptoms?\b", re.I),
    # "treat" and "cure" also mean a sweet snack and preserving meat, so they
    # refuse only when aimed at a condition.
    re.compile(rf"\b(?:treat(?:s|ing|ment)?|cur(?:e|es|ing))\b[^.?!]{{0,30}}?\b{_CONDITION}\b", re.I),
    # Third-person and population framing: "a diabetic", "people with diabetes"
    re.compile(
        r"\b(?:an? diabetic|diabetics|diabetic (?:people|persons?|patients?|adults|children|kids|women|men))\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:people|persons?|someone|anyone|those|patients?|adults|children|kids|women|men) "
        rf"(?:with|who have|living with|suffering from) {_CONDITION}\b",
        re.I,
    ),
    re.compile(
        r"\b(?:(?:during|in) pregnancy|(?:while|when|if) pregnant|"
        r"pregnant (?:women|woman|mothers?|people|ladies)|expect(?:ant|ing) mothers?)\b",
        re.I,
    ),
    re.compile(
        rf"\bmy (?:mom|mum|mother|dad|father|wife|husband|partner|child|kid|son|daughter|baby) "
        rf"(?:is|has|was|got)\b[^.?!]{{0,30}}?\b{_CONDITION}\b",
        re.I,
    ),
    # "what should a diabetic eat", "what can a pregnant woman eat"
    re.compile(
        rf"\bwhat (?:should|can|could) (?:a|an) (?:\w+ ){{0,2}}?{_CONDITION}\b[^.?!]{{0,20}}?\b(?:eat|drink|have)\b",
        re.I,
    ),
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
    for pattern in _MEDICAL_PATTERNS + _PERSONAL_HEALTH_PATTERNS + _WEIGHT_CALORIE_PATTERNS:
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

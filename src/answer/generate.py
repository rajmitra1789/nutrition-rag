"""Grounded answers from retrieved chunks. One document per claim block."""

from __future__ import annotations

import time
from collections import defaultdict

from groq import Groq, RateLimitError

from src.answer.guardrails import Refusal, classify_scope, not_in_corpus_refusal
from src.config import (
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
    MAX_DOCS_IN_ANSWER,
    PASSAGE_SCORE_GAP,
    PASSAGE_TOKEN_BUDGET,
    groq_api_key,
)
from src.retrieve.index import Hit, RetrievalResult, retrieve

RATE_LIMIT_MESSAGE = "Try again in a minute."


class _StillLimited(Exception):
    """The single 429 retry also failed."""


NOT_COVERED = "NOT_COVERED"


def _system_prompt(hit: Hit) -> str:
    """Name one document. A factual claim must end with that document's short tag."""
    return (
        f"Answer only from the passages for {hit.document_name} "
        f"({hit.publisher}, {hit.year}). "
        "First decide whether the passages directly answer the question. "
        f"If they do not, reply with exactly {NOT_COVERED} and nothing else. "
        "Never summarize passages that do not answer the question. "
        "If the question asks for an amount, number, duration or limit and the passages "
        "do not state that quantity for the exact thing asked about, "
        f"reply exactly {NOT_COVERED}. Other numbers in the passages do not count. "
        "Use only these passages. No outside knowledge. "
        "No medical advice, personal calorie targets, or weight targets. "
        "Use at most four short sentences. "
        "State each food's exact time from its own row and never merge ranges from different rows. "
        f"End every factual claim with {_short_citation(hit)}."
    )


def _citation(hit: Hit) -> str:
    """Full citation with the URL. Used once, in the section header built in code."""
    return f"({hit.document_name}, {hit.publisher}, {hit.year}, {hit.source_url})"


def _short_citation(hit: Hit) -> str:
    """Per-claim tag the model writes inline. The URL lives in the header."""
    return f"({hit.document_name}, {hit.publisher}, {hit.year})"


def _group_by_document(hits: list[Hit]) -> list[tuple[str, list[Hit]]]:
    grouped: dict[str, list[Hit]] = defaultdict(list)
    order: list[str] = []
    for hit in hits:
        if hit.doc_id not in grouped:
            order.append(hit.doc_id)
        grouped[hit.doc_id].append(hit)
    ranked = sorted(order, key=lambda doc_id: max(h.similarity for h in grouped[doc_id]), reverse=True)
    return [(doc_id, grouped[doc_id]) for doc_id in ranked[:MAX_DOCS_IN_ANSWER]]


def _estimate_tokens(text: str) -> int:
    """Same whitespace heuristic as scripts/analyze_chunks.py."""
    return max(1, int(round(len(text.split()) / 0.75)))


def passages_for_call(hits: list[Hit]) -> list[Hit]:
    """Best chunk, plus near-tied chunks that still fit the passage budget."""
    ranked = sorted(hits, key=lambda hit: hit.similarity, reverse=True)
    chosen = [ranked[0]]
    used = _estimate_tokens(ranked[0].text)
    best = ranked[0].similarity
    for hit in ranked[1:]:
        if best - hit.similarity > PASSAGE_SCORE_GAP:
            break
        extra = _estimate_tokens(hit.text)
        if used + extra > PASSAGE_TOKEN_BUDGET:
            break
        chosen.append(hit)
        used += extra
    return chosen


def _passages_block(hits: list[Hit]) -> str:
    body = "\n\n".join(f"{hit.section_heading}\n{hit.text}" for hit in hits)
    return f"{body}\n\nCite as: {_short_citation(hits[0])}"


def _client() -> Groq:
    key = groq_api_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or Streamlit secrets.")
    return Groq(api_key=key)


def _retry_after_seconds(exc: RateLimitError) -> float:
    raw = exc.response.headers.get("retry-after", "").strip()
    try:
        seconds = float(raw)
    except ValueError:
        seconds = 5.0
    return min(max(seconds, 1.0), 60.0)


def _complete(messages: list[dict[str, str]]):
    kwargs = {
        "model": GROQ_MODEL,
        "temperature": 0.1,
        "max_tokens": GROQ_MAX_TOKENS,
        "reasoning_effort": GROQ_REASONING_EFFORT,
        "messages": messages,
    }
    client = _client()
    try:
        return client.chat.completions.create(**kwargs)
    except RateLimitError as exc:
        time.sleep(_retry_after_seconds(exc))
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as retry_exc:
            raise _StillLimited from retry_exc


def _generate_for_document(query: str, hits: list[Hit]) -> str:
    passages = passages_for_call(hits)
    head = passages[0]
    user = f"Question: {query}\n\n{_passages_block(passages)}"
    response = _complete(
        [
            {"role": "system", "content": _system_prompt(head)},
            {"role": "user", "content": user},
        ]
    )
    return (response.choices[0].message.content or "").strip()


def _heading_for(hits: list[Hit]) -> str:
    hit = max(hits, key=lambda item: item.similarity)
    return f"**{hit.document_name}** — {hit.publisher} ({hit.year}) {_citation(hit)}"


def is_not_covered(text: str) -> bool:
    """True when the reply is empty or contains NOT_COVERED anywhere."""
    cleaned = (text or "").strip()
    return cleaned == "" or NOT_COVERED in cleaned


def _section(hits: list[Hit], body: str) -> str | None:
    if is_not_covered(body):
        return None
    return f"{_heading_for(hits)}\n\n{body.strip()}"


def answer_from_retrieval(query: str, result: RetrievalResult) -> str | None:
    """Turn retrieval hits into a per-document answer. Never blends sources.

    Returns None when every section only declines, so the caller can refuse.
    """
    groups = _group_by_document(result.hits)
    try:
        sections: list[str] = []
        for _, hits in groups:
            section = _section(hits, _generate_for_document(query, hits))
            if section:
                sections.append(section)
        if not sections:
            return None
        return "\n\n---\n\n".join(sections)
    except _StillLimited:
        return RATE_LIMIT_MESSAGE


def answer_question(query: str, doc_id: str | None = None) -> tuple[str, RetrievalResult | None, Refusal | None]:
    """
    Full answer path with code-enforced refusals.

    Returns (message, retrieval, refusal). Retrieval is None when the query is
    refused as out of scope before search.
    """
    scoped = classify_scope(query)
    if scoped:
        return scoped.message, None, scoped

    result = retrieve(query, doc_id=doc_id)
    if not result.enough:
        refusal = not_in_corpus_refusal(result.searched_documents)
        return refusal.message, result, refusal

    text = answer_from_retrieval(query, result)
    if text is None:
        refusal = not_in_corpus_refusal(result.searched_documents)
        return refusal.message, result, refusal
    return text, result, None

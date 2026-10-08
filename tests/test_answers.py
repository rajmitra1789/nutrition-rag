"""Short inline tags, full citation in the header, and NOT_COVERED sections dropped."""

from types import SimpleNamespace

from src.answer.generate import (
    NOT_COVERED,
    _group_by_document,
    _heading_for,
    _passages_block,
    _system_prompt,
    answer_from_retrieval,
    answer_question,
    is_not_covered,
    passages_for_call,
)
from src.config import GROQ_MAX_TOKENS, MAX_DOCS_IN_ANSWER, PASSAGE_TOKEN_BUDGET, TOP_K
from src.retrieve.index import Hit, RetrievalResult


def _hit(
    doc_id: str,
    name: str,
    publisher: str,
    year: int,
    url: str,
    similarity: float = 0.9,
) -> Hit:
    return Hit(
        chunk_id=f"{doc_id}-001",
        doc_id=doc_id,
        document_name=name,
        publisher=publisher,
        year=year,
        source_url=url,
        section_heading="Section",
        text="passage",
        similarity=similarity,
    )


EGGS = _hit(
    "foodsafety_cold_storage",
    "Cold Food Storage Charts",
    "FoodSafety.gov (U.S. government)",
    2023,
    "https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts",
    0.85,
)
CANADA = _hit(
    "health_canada_recommendations",
    "Healthy eating recommendations",
    "Health Canada",
    2019,
    "https://www.canada.ca/en/health-canada/services/food-guide/explore/healthy-eating-recommendations.html",
    0.8,
)
EATWELL = _hit(
    "eatwell_guide",
    "The Eatwell Guide",
    "Office for Health Improvement and Disparities (OHID), GOV.UK",
    2024,
    "https://www.gov.uk/government/publications/the-eatwell-guide",
    0.81,
)

SEARCHED = [
    {
        "document_name": EGGS.document_name,
        "publisher": EGGS.publisher,
        "year": EGGS.year,
    },
    {
        "document_name": CANADA.document_name,
        "publisher": CANADA.publisher,
        "year": CANADA.year,
    },
]


def test_section_header_includes_citation_from_hit_metadata():
    heading = _heading_for([EGGS])
    assert heading.endswith(
        "(Cold Food Storage Charts, FoodSafety.gov (U.S. government), 2023, "
        "https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts)"
    )


# --- Issue 1: short inline tags, URL only in the header, enough token room ---


def test_prompt_asks_for_short_tag_without_url():
    prompt = _system_prompt(EATWELL)
    assert (
        "End every factual claim with (The Eatwell Guide, "
        "Office for Health Improvement and Disparities (OHID), GOV.UK, 2024)." in prompt
    )
    assert "http" not in prompt


def test_passage_block_shows_short_tag_without_url():
    block = _passages_block([EGGS])
    assert block.endswith(
        "Cite as: (Cold Food Storage Charts, FoodSafety.gov (U.S. government), 2023)"
    )
    assert "http" not in block


def test_url_appears_once_per_section_in_the_header(monkeypatch):
    monkeypatch.setattr(
        "src.answer.generate._generate_for_document",
        lambda query, hits: (
            "Raw eggs last 3 to 5 weeks (Cold Food Storage Charts, FoodSafety.gov (U.S. government), 2023). "
            "Hard-cooked eggs last 1 week (Cold Food Storage Charts, FoodSafety.gov (U.S. government), 2023)."
        ),
    )
    result = RetrievalResult(hits=[EGGS], searched_documents=SEARCHED, enough=True)
    text = answer_from_retrieval("How long do eggs keep?", result)
    assert text is not None
    assert text.count(EGGS.source_url) == 1
    assert text.splitlines()[0].endswith(f"{EGGS.source_url})")


def _fake_groq(reply: str, captured: dict):
    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_each_call_uses_the_configured_token_cap(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr("src.answer.generate._client", lambda: _fake_groq("Answer.", captured))
    result = RetrievalResult(hits=[EGGS], searched_documents=SEARCHED, enough=True)
    answer_from_retrieval("How long do eggs keep?", result)
    assert captured["max_tokens"] == GROQ_MAX_TOKENS
    assert GROQ_MAX_TOKENS >= 320
    assert captured["reasoning_effort"] == "low"


# --- Issue 2: NOT_COVERED sections are dropped; all dropped means refusal ---


def test_is_not_covered_matches_the_marker_anywhere():
    assert is_not_covered(NOT_COVERED)
    assert is_not_covered("  NOT_COVERED\n")
    assert is_not_covered("NOT_COVERED.")
    assert is_not_covered("NOT_COVERED (Healthy eating recommendations, Health Canada, 2019)")
    assert is_not_covered("")
    assert is_not_covered("Raw eggs last 3 to 5 weeks. NOT_COVERED")
    assert is_not_covered("Oil can be reused a few times.\n\nNOT_COVERED")
    assert not is_not_covered("Raw eggs last 3 to 5 weeks.")


def test_prompt_decides_coverage_before_answering():
    prompt = _system_prompt(EGGS)
    decide = "First decide whether the passages directly answer the question."
    refuse = "If they do not, reply with exactly NOT_COVERED and nothing else."
    assert decide in prompt
    assert refuse in prompt
    assert "Never summarize passages that do not answer the question." in prompt
    assert prompt.index(decide) < prompt.index(refuse) < prompt.index("Use at most four short sentences.")


def test_prompt_requires_a_stated_quantity_for_quantity_questions():
    prompt = _system_prompt(EGGS)
    rule = (
        "If the question asks for an amount, number, duration or limit and the passages "
        "do not state that quantity for the exact thing asked about, "
        "reply exactly NOT_COVERED. Other numbers in the passages do not count."
    )
    assert rule in prompt
    assert prompt.index(rule) < prompt.index("Use at most four short sentences.")


def test_summary_ending_in_not_covered_is_dropped_and_marker_never_shown(monkeypatch):
    def fake_generate(query: str, hits: list[Hit]) -> str:
        if hits[0].doc_id == EGGS.doc_id:
            return "Cook poultry to 165 °F (Cold Food Storage Charts, FoodSafety.gov (U.S. government), 2023)."
        return "Reused oil should be discarded when it smells rancid (Healthy eating recommendations, Health Canada, 2019). NOT_COVERED"

    monkeypatch.setattr("src.answer.generate._generate_for_document", fake_generate)
    result = RetrievalResult(hits=[EGGS, CANADA], searched_documents=SEARCHED, enough=True)
    text = answer_from_retrieval("What is the safe internal temperature for chicken?", result)
    assert text is not None
    assert NOT_COVERED not in text
    assert "Reused oil" not in text
    assert CANADA.document_name not in text
    assert "165 °F" in text


def test_not_covered_reply_from_groq_drops_that_section(monkeypatch):
    def client():
        return _fake_groq(NOT_COVERED, {})

    monkeypatch.setattr("src.answer.generate._client", client)
    result = RetrievalResult(hits=[CANADA], searched_documents=SEARCHED, enough=True)
    assert answer_from_retrieval("How long do eggs keep?", result) is None


def test_prompt_forbids_merging_storage_rows():
    prompt = _system_prompt(EGGS)
    assert (
        "State each food's exact time from its own row and never merge ranges from different rows."
        in prompt
    )


def test_mixed_sections_keep_the_answer_and_its_header_link(monkeypatch):
    def fake_generate(query: str, hits: list[Hit]) -> str:
        if hits[0].doc_id == EGGS.doc_id:
            return "Raw eggs in the shell last 3 to 5 weeks."
        return NOT_COVERED

    monkeypatch.setattr("src.answer.generate._generate_for_document", fake_generate)
    result = RetrievalResult(hits=[EGGS, CANADA], searched_documents=SEARCHED, enough=True)
    text = answer_from_retrieval("How long can I keep eggs in the fridge?", result)
    assert text is not None
    assert "Raw eggs in the shell last 3 to 5 weeks." in text
    assert EGGS.source_url in text
    assert NOT_COVERED not in text
    assert "---" not in text
    assert CANADA.document_name not in text


def test_every_declined_section_returns_not_in_corpus(monkeypatch):
    monkeypatch.setattr(
        "src.answer.generate.retrieve",
        lambda *args, **kwargs: RetrievalResult(
            hits=[EGGS, CANADA],
            searched_documents=SEARCHED,
            enough=True,
        ),
    )
    monkeypatch.setattr(
        "src.answer.generate._generate_for_document",
        lambda *args, **kwargs: NOT_COVERED,
    )
    message, result, refusal = answer_question("How long can I keep eggs in the fridge?")
    assert refusal is not None
    assert refusal.kind == "not_in_corpus"
    assert result is not None
    assert "Cold Food Storage Charts" in message
    assert "FoodSafety.gov (U.S. government)" in message
    assert "Healthy eating recommendations" in message
    assert "Health Canada" in message


# --- Cooking oil per day: the ICMR plate table must reach its call ---


def test_retrieval_and_answer_limits():
    assert TOP_K == 10
    assert MAX_DOCS_IN_ANSWER == 5
    assert PASSAGE_TOKEN_BUDGET == 450


def _sized_hit(chunk_id: str, doc_id: str, similarity: float, tokens: int) -> Hit:
    words = " ".join(["w"] * round(tokens * 0.75))
    return Hit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        document_name=doc_id,
        publisher="Publisher",
        year=2024,
        source_url=f"https://example.org/{doc_id}",
        section_heading="Section",
        text=words,
        similarity=similarity,
    )


def test_icmr_plate_table_fits_the_passage_budget():
    best = _sized_hit("icmr_nin_my_plate-002", "icmr_nin_my_plate", 0.765, 153)
    table = _sized_hit("icmr_nin_my_plate-004", "icmr_nin_my_plate", 0.762, 269)
    assert [hit.chunk_id for hit in passages_for_call([best, table])] == [
        "icmr_nin_my_plate-002",
        "icmr_nin_my_plate-004",
    ]


def test_fifth_document_is_answered():
    hits = [
        _sized_hit("sfa-002", "sfa", 0.827, 100),
        _sized_hit("eatwell-007", "eatwell", 0.791, 100),
        _sized_hit("who-005", "who", 0.782, 100),
        _sized_hit("who_sfa-006", "who_sfa", 0.766, 100),
        _sized_hit("icmr_nin_my_plate-002", "icmr_nin_my_plate", 0.765, 100),
    ]
    assert [doc_id for doc_id, _ in _group_by_document(hits)][-1] == "icmr_nin_my_plate"

"""Guardrails must fire in code, not only in the model prompt."""

from src.answer.guardrails import classify_scope, not_in_corpus_refusal


def test_refuses_personal_calorie_target():
    refusal = classify_scope("How many calories should I eat to lose weight?")
    assert refusal is not None
    assert refusal.kind == "out_of_scope"
    assert "qualified professional" in refusal.message.lower()


def test_refuses_weight_target():
    refusal = classify_scope("What should I weigh?")
    assert refusal is not None
    assert refusal.kind == "out_of_scope"


def test_refuses_medical_advice():
    refusal = classify_scope("Can you diagnose my symptoms and tell me what medication to take?")
    assert refusal is not None
    assert refusal.kind == "out_of_scope"


def test_allows_document_questions_about_fat_and_oil():
    assert classify_scope("What does official guidance say about reusing cooking oil?") is None
    assert classify_scope("What does WHO recommend for saturated fat as a share of energy?") is None
    assert classify_scope("How long can leftover pizza stay in the fridge?") is None


def test_unmatched_phrasing_is_not_refused_by_the_regex():
    """The regex list is fixed. Phrasings it misses still reach retrieval.

    The system prompt also forbids medical advice. That line is not the enforcement.
    """
    assert classify_scope("I have diabetes, what should I eat?") is None


def test_not_in_corpus_names_searched_documents():
    docs = [
        {
            "document_name": "Reusing Cooking Oils",
            "publisher": "Singapore Food Agency",
            "year": 2024,
        }
    ]
    refusal = not_in_corpus_refusal(docs)
    assert refusal.kind == "not_in_corpus"
    assert "Reusing Cooking Oils" in refusal.message
    assert "Singapore Food Agency" in refusal.message

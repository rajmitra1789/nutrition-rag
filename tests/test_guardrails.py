"""Guardrails must fire in code, not only in the model prompt."""

import pytest

from src.answer.guardrails import OUT_OF_SCOPE_MESSAGE, classify_scope, not_in_corpus_refusal


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


@pytest.mark.parametrize(
    "query",
    [
        "I have diabetes, what should I eat?",
        "I am pregnant, is it safe to eat raw eggs?",
        "what can I eat to cure high blood pressure",
        "my doctor prescribed a low salt diet, what should I eat",
    ],
)
def test_refuses_personal_health_conditions(query):
    refusal = classify_scope(query)
    assert refusal is not None
    assert refusal.kind == "out_of_scope"
    assert refusal.message == OUT_OF_SCOPE_MESSAGE


@pytest.mark.parametrize(
    "query",
    [
        "What should a diabetic eat?",
        "Is raw egg safe during pregnancy?",
        "My mom is diabetic, what can she eat?",
        "What can pregnant women eat?",
    ],
)
def test_refuses_third_person_and_population_health_framing(query):
    refusal = classify_scope(query)
    assert refusal is not None
    assert refusal.kind == "out_of_scope"
    assert refusal.message == OUT_OF_SCOPE_MESSAGE


@pytest.mark.parametrize(
    "query",
    [
        "How long can I keep kidney beans?",
        "Is a sweet treat ok for kids?",
        "How do you cure ham safely?",
        "How long can I keep eggs in the fridge?",
        "What does the guidance say about cooking oil?",
        "What does WHO say about salt?",
        "What does WHO say about sodium?",
    ],
)
def test_general_questions_pass_through(query):
    assert classify_scope(query) is None


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

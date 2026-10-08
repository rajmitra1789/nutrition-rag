"""Print kept chunks and per-call Groq token usage for one question.

Usage: python scripts/token_report.py ["question"]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.answer.generate as generate  # noqa: E402
from src.config import GROQ_MAX_TOKENS  # noqa: E402

DEFAULT_QUESTION = "What does the guidance say about cooking oil?"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    calls: list[dict] = []
    sent: list[list[str]] = []
    original_complete = generate._complete
    original_passages = generate.passages_for_call

    def recording_passages(hits):
        chosen = original_passages(hits)
        sent.append([hit.chunk_id for hit in chosen])
        return chosen

    def recording_complete(messages):
        response = original_complete(messages)
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        calls.append(
            {
                "system": messages[0]["content"],
                "finish_reason": response.choices[0].finish_reason,
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
                "reasoning": getattr(details, "reasoning_tokens", None) if details else None,
                "content": response.choices[0].message.content or "",
            }
        )
        return response

    generate.passages_for_call = recording_passages
    generate._complete = recording_complete
    message, result, refusal = generate.answer_question(question)

    print(f"Question: {question}")
    print(f"GROQ_MAX_TOKENS: {GROQ_MAX_TOKENS}")
    print(f"Refusal: {refusal.kind if refusal else 'none'}\n")
    if result:
        print("Retrieved:")
        for hit in result.hits:
            print(f"  {hit.chunk_id:32} {hit.similarity:.3f}")
        print()
    for index, call in enumerate(calls, 1):
        name = call["system"].split("passages for ", 1)[-1].split(" (", 1)[0]
        verdict = "DROPPED" if generate.is_not_covered(call["content"]) else "KEPT"
        print(
            f"Call {index} [{name}] {verdict} sent={sent[index - 1]} "
            f"finish={call['finish_reason']} prompt={call['prompt']} "
            f"completion={call['completion']} reasoning={call['reasoning']}"
        )
    print(f"\nTotal tokens: {sum(call['total'] for call in calls)}")
    print("\n" + message)


if __name__ == "__main__":
    main()

"""Measure chunk lengths and record the embedding-model choice."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ANALYSIS_PATH, CHUNKS_PATH, EMBEDDING_MODEL

CHOSEN_MODEL = "jinaai/jina-embeddings-v2-small-en"
CHOSEN_WINDOW = 8192


def estimate_tokens(text: str) -> int:
    """Whitespace heuristic: ~0.75 words per token for official English prose."""
    words = len(text.split())
    return max(1, int(round(words / 0.75)))


def main() -> None:
    chunks = json.loads(Path(CHUNKS_PATH).read_text(encoding="utf-8"))
    chars = [c["char_count"] for c in chunks]
    tokens = [estimate_tokens(c["text"]) for c in chunks]
    protected = sum(1 for c in chunks if c["protected"])
    over_256 = sum(1 for t in tokens if t > 256)
    over_512 = sum(1 for t in tokens if t > 512)
    over_2048 = sum(1 for t in tokens if t > 2048)
    over_8192 = sum(1 for t in tokens if t > 8192)
    max_tokens = max(tokens)

    over_512_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "section_heading": chunk["section_heading"],
            "char_count": chunk["char_count"],
            "estimated_tokens": estimate_tokens(chunk["text"]),
            "protected": chunk["protected"],
        }
        for chunk in chunks
        if estimate_tokens(chunk["text"]) > 512
    ]

    analysis = {
        "n_chunks": len(chunks),
        "protected_units": protected,
        "by_document": dict(Counter(chunk["doc_id"] for chunk in chunks)),
        "chars": {
            "min": min(chars),
            "median": int(statistics.median(chars)),
            "mean": round(statistics.mean(chars), 1),
            "p95": int(statistics.quantiles(chars, n=20)[18]),
            "max": max(chars),
        },
        "estimated_tokens": {
            "min": min(tokens),
            "median": int(statistics.median(tokens)),
            "mean": round(statistics.mean(tokens), 1),
            "p95": int(statistics.quantiles(tokens, n=20)[18]),
            "max": max_tokens,
            "over_256": over_256,
            "over_512": over_512,
            "over_2048": over_2048,
            "over_8192": over_8192,
        },
        "over_512_chunks": over_512_chunks,
        "embedding_choice": {
            "model": CHOSEN_MODEL,
            "window_tokens": CHOSEN_WINDOW,
            "dimensions": 512,
            "onnx_gb": 0.12,
            "query_prefix": "",
            "fits_window": max_tokens <= CHOSEN_WINDOW,
            "configured_model": EMBEDDING_MODEL,
            "reason": (
                f"Measured max is {max_tokens} estimated tokens across {len(chunks)} chunks "
                f"({protected} protected). all-MiniLM-L6-v2 (256) would truncate {over_256} chunks. "
                f"bge-small-en-v1.5 (512, 33M, 384-d) would truncate {over_512} chunks, including "
                "the WHO SFA/TFA Background section. jina-embeddings-v2-small-en (8192 tokens, "
                "33M, 512-d, 0.12 GB ONNX) covers every chunk and uses no query prefix. "
                "Served via FastEmbed so Streamlit Community Cloud does not load PyTorch."
            ),
            "rejected": [
                {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "window_tokens": 256,
                    "why": (
                        f"256-token window truncates {over_256} chunks, including protected tables."
                    ),
                },
                {
                    "model": "BAAI/bge-small-en-v1.5",
                    "window_tokens": 512,
                    "why": (
                        f"512-token window truncates {over_512} unprotected paragraphs. "
                        "The longest would lose about 380 of 893 tokens."
                    ),
                },
                {
                    "model": "snowflake/snowflake-arctic-embed-m-long",
                    "window_tokens": 2048,
                    "why": (
                        "2048-token window fits, but the ONNX file is 0.54 GB and queries "
                        "need a query: prefix."
                    ),
                },
                {
                    "model": "nomic-ai/nomic-embed-text-v1.5",
                    "window_tokens": 8192,
                    "why": (
                        "8192-token window fits, but passages must be stored with a "
                        "search_document: prefix and the ONNX file is 0.52 GB."
                    ),
                },
            ],
        },
    }
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2))
    if EMBEDDING_MODEL != CHOSEN_MODEL:
        print(
            f"warning: EMBEDDING_MODEL is {EMBEDDING_MODEL}, analysis chose {CHOSEN_MODEL}",
            file=sys.stderr,
        )
    if not analysis["embedding_choice"]["fits_window"]:
        raise SystemExit(
            f"longest chunk is {max_tokens} estimated tokens, over the {CHOSEN_WINDOW} window"
        )


if __name__ == "__main__":
    main()

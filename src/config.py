"""Shared paths and model settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "src" / "corpus" / "manifest.json"
EXTRACTED_DIR = ROOT / "data" / "processed" / "extracted"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
ANALYSIS_PATH = ROOT / "data" / "processed" / "chunk_analysis.json"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "dietary_guidance"

# Chosen in scripts/analyze_chunks.py from measured chunk lengths.
# jina-embeddings-v2-small-en is a 33M-parameter English retriever with an
# 8,192-token window, so the longest chunk (~893 estimated tokens) is embedded
# whole. It does not use a query or passage prefix.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-small-en")
EMBEDDING_QUERY_PREFIX = ""

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Free-plan caps for openai/gpt-oss-120b: 30 requests/minute, 1,000/day,
# 8,000 tokens/minute, 200,000 tokens/day. max_tokens covers reasoning plus
# the answer. reasoning_effort stays low so a short cap still leaves an answer.
# 320 is the longest measured completion on the cooking-oil question (252, SFA
# section, 11 uncapped runs with short inline tags) plus about 25% headroom.
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "320"))
GROQ_REASONING_EFFORT = "low"
# Extra passages join a call only when they sit this close to that document's
# best hit and the passage text stays inside this estimated-token budget.
PASSAGE_SCORE_GAP = 0.01
# 450 fits the ICMR best chunk (~153) plus the My Plate table (~269), which
# holds the Fats & Oils 27 g/day row.
PASSAGE_TOKEN_BUDGET = 450

# Cosine similarity floor. Hits below this are treated as not-in-corpus.
# 0.75 sits above off-topic probes on this index (wifi 0.722) and below
# in-corpus tops (weakest measured top was 0.827).
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.75"))
TOP_K = int(os.getenv("TOP_K", "10"))
# Per document, after the floor: at most this many chunks, and only those
# within PER_DOC_WINDOW of that document's best score. No cross-document margin.
PER_DOC_CHUNK_CAP = 3
PER_DOC_WINDOW = 0.04
MAX_DOCS_IN_ANSWER = int(os.getenv("MAX_DOCS_IN_ANSWER", "5"))

# Chunker targets. Protected units (tables, numbered recommendation sets)
# are never split to meet these numbers.
TARGET_CHUNK_CHARS = 1600
MAX_CHUNK_CHARS = 2400


def groq_api_key() -> str:
    """Read the Groq key from the environment or Streamlit secrets."""
    if GROQ_API_KEY:
        return GROQ_API_KEY
    try:
        import streamlit as st

        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return ""

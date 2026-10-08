"""Embed chunks and persist the ChromaDB collection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieve.index import build_index


def main() -> None:
    count = build_index()
    print(f"Indexed {count} chunks")


if __name__ == "__main__":
    main()

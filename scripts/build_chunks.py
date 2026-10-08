"""Chunk the extracted corpus and write data/processed/chunks.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHUNKS_PATH
from src.ingest.chunk import build_chunks


def main() -> None:
    chunks = build_chunks()
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    by_doc: dict[str, int] = {}
    protected = 0
    for chunk in chunks:
        by_doc[chunk["doc_id"]] = by_doc.get(chunk["doc_id"], 0) + 1
        if chunk["protected"]:
            protected += 1
    print(f"Wrote {len(chunks)} chunks to {CHUNKS_PATH}")
    print(f"Protected tables/numbered blocks: {protected}")
    for doc_id, count in by_doc.items():
        print(f"  {doc_id}: {count}")


if __name__ == "__main__":
    main()

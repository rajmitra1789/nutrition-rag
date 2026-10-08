"""Print a few chunk embeddings already stored in the Chroma collection.

Opens the persistent index read-only. Does not load the embedding model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME


def _preview(values: list[float], head: int) -> str:
    shown = ", ".join(f"{value:.4f}" for value in values[:head])
    if len(values) > head:
        return f"[{shown}, …]  ({len(values)} dims)"
    return f"[{shown}]  ({len(values)} dims)"


def _snippet(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="how many chunks to print")
    parser.add_argument(
        "--dims",
        type=int,
        default=8,
        help="how many leading embedding values to print",
    )
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    total = collection.count()
    limit = max(1, min(args.n, total))

    rows = collection.get(
        limit=limit,
        include=["embeddings", "documents", "metadatas"],
    )

    ids = rows["ids"]
    embeddings = rows["embeddings"]
    documents = rows["documents"]
    metadatas = rows["metadatas"]

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Path: {CHROMA_DIR}")
    print(f"Stored chunks: {total}")
    print(f"Showing: {len(ids)}")
    print()

    for index, chunk_id in enumerate(ids):
        vector = [float(value) for value in embeddings[index]]
        meta = metadatas[index] or {}
        text = documents[index] or ""
        norm = sum(value * value for value in vector) ** 0.5

        print(f"--- {chunk_id}")
        print(f"document:  {meta.get('document_name', '')}")
        print(f"section:   {meta.get('section_heading', '')}")
        print(f"publisher: {meta.get('publisher', '')} ({meta.get('year', '')})")
        print(f"text:      {_snippet(text)}")
        print(f"norm:      {norm:.4f}")
        print(f"embedding: {_preview(vector, args.dims)}")
        print()


if __name__ == "__main__":
    main()

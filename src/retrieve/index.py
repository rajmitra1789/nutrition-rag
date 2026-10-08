"""ChromaDB index build and query, with optional per-document filter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from src.config import (
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_PREFIX,
    MIN_SIMILARITY,
    PER_DOC_CHUNK_CAP,
    PER_DOC_WINDOW,
    TOP_K,
)
from src.ingest.chunk import load_manifest


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    section_heading: str
    text: str
    similarity: float


@dataclass
class RetrievalResult:
    hits: list[Hit]
    searched_documents: list[dict[str, Any]]
    enough: bool


class FastEmbedEmbeddingFunction(EmbeddingFunction[Documents]):
    """ONNX embedder via FastEmbed. Avoids loading PyTorch on Streamlit Community Cloud."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from fastembed import TextEmbedding

        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def name(self) -> str:
        return f"fastembed-{self._model_name}"

    def __call__(self, input: Documents) -> Embeddings:
        vectors = [list(map(float, vector)) for vector in self._model.embed(list(input))]
        return vectors


def embedding_function() -> FastEmbedEmbeddingFunction:
    return FastEmbedEmbeddingFunction(EMBEDDING_MODEL)


def _client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def load_chunks() -> list[dict[str, Any]]:
    return json.loads(Path(CHUNKS_PATH).read_text(encoding="utf-8"))


def build_index(chunks: list[dict[str, Any]] | None = None) -> int:
    """Rebuild the persistent Chroma collection from chunks.json."""
    records = chunks if chunks is not None else load_chunks()
    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    get_collection.cache_clear()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    ids = [row["chunk_id"] for row in records]
    documents = [row["text"] for row in records]
    metadatas = [
        {
            "doc_id": row["doc_id"],
            "document_name": row["document_name"],
            "publisher": row["publisher"],
            "year": int(row["year"]),
            "source_url": row["source_url"],
            "retrieval_date": str(row["retrieval_date"]),
            "section_heading": row["section_heading"],
            "protected": bool(row["protected"]),
        }
        for row in records
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    get_collection.cache_clear()
    return len(records)


@lru_cache(maxsize=1)
def get_collection():
    client = _client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def document_catalog() -> list[dict[str, Any]]:
    return load_manifest()["documents"]


def searched_documents(doc_id: str | None) -> list[dict[str, Any]]:
    catalog = document_catalog()
    if not doc_id:
        return catalog
    return [doc for doc in catalog if doc["doc_id"] == doc_id]


def _select_hits(hits: list[Hit], top_k: int) -> list[Hit]:
    """Floor, then at most three chunks within 0.04 of each document's best."""
    above = [hit for hit in hits if hit.similarity >= MIN_SIMILARITY]
    by_doc: dict[str, list[Hit]] = {}
    for hit in above:
        by_doc.setdefault(hit.doc_id, []).append(hit)

    kept: list[Hit] = []
    for group in by_doc.values():
        group.sort(key=lambda hit: hit.similarity, reverse=True)
        best = group[0].similarity
        chosen = [
            hit for hit in group if best - hit.similarity <= PER_DOC_WINDOW
        ][:PER_DOC_CHUNK_CAP]
        kept.extend(chosen)

    kept.sort(key=lambda hit: hit.similarity, reverse=True)
    return kept[:top_k]


def retrieve(query: str, doc_id: str | None = None, top_k: int = TOP_K) -> RetrievalResult:
    """Retrieve across all documents, or only the named document."""
    collection = get_collection()
    where = {"doc_id": doc_id} if doc_id else None
    query_text = f"{EMBEDDING_QUERY_PREFIX}{query}" if EMBEDDING_QUERY_PREFIX else query
    matched_ids = collection.get(where=where, include=[])["ids"]
    if not matched_ids:
        return RetrievalResult(
            hits=[],
            searched_documents=searched_documents(doc_id),
            enough=False,
        )

    raw = collection.query(
        query_texts=[query_text],
        n_results=len(matched_ids),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[Hit] = []
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
        similarity = 1.0 - float(distance)
        hits.append(
            Hit(
                chunk_id=chunk_id,
                doc_id=str(meta["doc_id"]),
                document_name=str(meta["document_name"]),
                publisher=str(meta["publisher"]),
                year=int(meta["year"]),
                source_url=str(meta["source_url"]),
                section_heading=str(meta["section_heading"]),
                text=text,
                similarity=similarity,
            )
        )

    relevant = _select_hits(hits, top_k)
    return RetrievalResult(
        hits=relevant,
        searched_documents=searched_documents(doc_id),
        enough=bool(relevant),
    )

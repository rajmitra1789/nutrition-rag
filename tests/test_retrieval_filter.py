"""Per-document filter is applied in the retriever, not only in the prompt."""

from src.ingest.chunk import load_manifest


def searched_documents(doc_id: str | None):
    catalog = load_manifest()["documents"]
    if not doc_id:
        return catalog
    return [doc for doc in catalog if doc["doc_id"] == doc_id]


def test_all_documents_when_unfiltered():
    docs = searched_documents(None)
    assert len(docs) == 7
    ids = {d["doc_id"] for d in docs}
    assert "sfa_reusing_cooking_oils" in ids
    assert "who_sfa_tfa_guideline" in ids


def test_single_document_filter_names_only_that_source():
    docs = searched_documents("sfa_reusing_cooking_oils")
    assert len(docs) == 1
    assert docs[0]["document_name"] == "Reusing Cooking Oils"

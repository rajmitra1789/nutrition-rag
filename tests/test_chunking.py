"""Tables and numbered recommendations must stay intact."""

from src.ingest.chunk import atomic_units, chunk_document, pack_units, split_sections

DOC = {
    "doc_id": "demo",
    "document_name": "Demo",
    "publisher": "Test Publisher",
    "year": 2024,
    "source_url": "https://example.org/demo",
    "retrieval_date": "2026-10-05",
}


def test_does_not_split_markdown_table():
    body = """Intro sentence.

| Food | Fridge |
| --- | --- |
| Pizza | 3 to 4 days |
| Chicken | 3 to 4 days |

Closing sentence.
"""
    units = atomic_units(body)
    protected = [text for text, flag in units if flag]
    assert len(protected) == 1
    assert "Pizza" in protected[0] and "Chicken" in protected[0]


def test_does_not_split_numbered_recommendations():
    body = """Context.

1. WHO recommends that adults and children reduce saturated fatty acid intake to 10% of total energy intake (strong recommendation).
2. WHO suggests further reducing saturated fatty acid intake to less than 10% of total energy intake (conditional recommendation).
3. WHO recommends replacing saturated fatty acids in the diet with polyunsaturated fatty acids (strong recommendation).

Remarks follow.
"""
    units = atomic_units(body)
    protected = [text for text, flag in units if flag]
    assert len(protected) == 1
    assert protected[0].startswith("1. ")
    assert "3. WHO recommends replacing" in protected[0]


def test_section_headings_are_preserved_on_chunks():
    markdown = """# Title

## Oil and spreads

Choose unsaturated oils and spreads and eat in small amounts.

## Fluids

Aim to drink 6-8 glasses of fluid every day.
"""
    chunks = chunk_document(DOC, markdown)
    headings = {c.section_heading for c in chunks}
    assert "Oil and spreads" in headings
    assert "Fluids" in headings
    assert all(c.retrieval_date == "2026-10-05" for c in chunks)


def test_retrieval_date_comes_from_extracted_frontmatter():
    markdown = """---
doc_id: demo
retrieval_date: 2026-10-05
---

## Fluids

Aim to drink 6-8 glasses of fluid every day.
"""
    doc = {key: value for key, value in DOC.items() if key != "retrieval_date"}
    chunks = chunk_document(doc, markdown)
    assert chunks
    assert all(chunk.retrieval_date == "2026-10-05" for chunk in chunks)


def test_pack_units_keeps_protected_block_whole():
    table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    packed = pack_units("Section", [("short intro", False), (table, True)])
    assert packed[-1] == (table, True)

"""Structure-aware chunking that never splits tables or numbered recommendations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import (
    EXTRACTED_DIR,
    MANIFEST_PATH,
    MAX_CHUNK_CHARS,
    TARGET_CHUNK_CHARS,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
NUMBERED_RE = re.compile(r"^\s*\d+\.\s+\S")
TABLE_ROW_RE = re.compile(r"^\s*\|")
FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
FRONTMATTER_FIELD_RE = re.compile(r"^([A-Za-z0-9_]+):\s*(.*?)\s*$")


def parse_frontmatter(markdown: str) -> dict[str, str]:
    """Read YAML fields from an extracted markdown file. Body text is ignored."""
    if not markdown.startswith("---"):
        return {}
    match = FRONTMATTER_RE.match(markdown)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(0).splitlines()[1:-1]:
        found = FRONTMATTER_FIELD_RE.match(line)
        if not found:
            continue
        fields[found.group(1)] = _unquote_frontmatter(found.group(2))
    return fields


def _unquote_frontmatter(value: str) -> str:
    value = value.strip()
    if value.startswith('"'):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return value.strip('"')
        return loaded if isinstance(loaded, str) else str(loaded)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


def strip_frontmatter(markdown: str) -> str:
    """Drop YAML document metadata so it is not chunked as prose."""
    if markdown.startswith("---"):
        match = FRONTMATTER_RE.match(markdown)
        if match:
            return markdown[match.end() :]
    return markdown


def retrieval_date_for(doc: dict, markdown: str, fallback: str = "") -> str:
    """Date from the extracted file, then the catalog row, then the manifest root."""
    from_file = parse_frontmatter(markdown).get("retrieval_date", "").strip()
    if from_file:
        return from_file
    from_doc = str(doc.get("retrieval_date") or "").strip()
    if from_doc:
        return from_doc
    return fallback.strip()


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    retrieval_date: str
    section_heading: str
    text: str
    protected: bool
    char_count: int


def load_manifest() -> dict:
    return json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown document into (heading, body) pairs."""
    lines = markdown.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Document"
    current_body: list[str] = []

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            if current_body and any(part.strip() for part in current_body):
                sections.append((current_heading, current_body))
            current_heading = match.group(2).strip()
            current_body = []
            continue
        current_body.append(line)

    if current_body and any(part.strip() for part in current_body):
        sections.append((current_heading, current_body))
    return [(heading, "\n".join(body).strip()) for heading, body in sections]


def _is_table_line(line: str) -> bool:
    return bool(TABLE_ROW_RE.match(line))


def _is_numbered_line(line: str) -> bool:
    return bool(NUMBERED_RE.match(line))


def atomic_units(body: str) -> list[tuple[str, bool]]:
    """
    Break a section into atomic units.

    A unit is protected when it is a markdown table or a consecutive numbered
    recommendation list. Protected units are never split later.
    """
    lines = body.splitlines()
    units: list[tuple[str, bool]] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if _is_table_line(line):
            block = [line]
            i += 1
            while i < n and (_is_table_line(lines[i]) or not lines[i].strip()):
                if lines[i].strip():
                    block.append(lines[i])
                i += 1
            units.append(("\n".join(block).strip(), True))
            continue

        if _is_numbered_line(line):
            block = [line]
            i += 1
            while i < n:
                nxt = lines[i]
                if _is_numbered_line(nxt) or (nxt.startswith(" ") and nxt.strip()):
                    block.append(nxt)
                    i += 1
                    continue
                if not nxt.strip():
                    # Allow a blank line inside a numbered set only if the next
                    # non-empty line is still numbered.
                    look = i + 1
                    while look < n and not lines[look].strip():
                        look += 1
                    if look < n and _is_numbered_line(lines[look]):
                        i = look
                        continue
                break
            units.append(("\n".join(block).strip(), True))
            continue

        if not line.strip():
            i += 1
            continue

        para = [line]
        i += 1
        while i < n and lines[i].strip() and not _is_table_line(lines[i]) and not _is_numbered_line(lines[i]) and not HEADING_RE.match(lines[i]):
            if lines[i].lstrip().startswith("- ") and para and not para[-1].lstrip().startswith("- "):
                break
            para.append(lines[i])
            i += 1
        units.append(("\n".join(para).strip(), False))

    return [(text, protected) for text, protected in units if text]


def pack_units(heading: str, units: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    """Pack non-protected units up to TARGET_CHUNK_CHARS. Protected units stay whole."""
    packed: list[tuple[str, bool]] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if buffer:
            packed.append(("\n\n".join(buffer).strip(), False))
            buffer = []
            buffer_len = 0

    for text, protected in units:
        if protected:
            flush()
            packed.append((text, True))
            continue

        extra = len(text) + (2 if buffer else 0)
        if buffer and buffer_len + extra > TARGET_CHUNK_CHARS:
            flush()
        if len(text) > MAX_CHUNK_CHARS:
            # Long prose only: split on paragraph boundaries, never on tables.
            for piece in _split_long_prose(text):
                packed.append((piece, False))
            continue
        buffer.append(text)
        buffer_len += extra

    flush()
    return packed


def _split_long_prose(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:
        return [text]
    pieces: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paragraphs:
        extra = len(para) + (2 if buf else 0)
        if buf and size + extra > TARGET_CHUNK_CHARS:
            pieces.append("\n\n".join(buf))
            buf = [para]
            size = len(para)
        else:
            buf.append(para)
            size += extra
    if buf:
        pieces.append("\n\n".join(buf))
    return pieces


def chunk_document(doc: dict, markdown: str, *, fallback_retrieval_date: str = "") -> list[Chunk]:
    retrieval_date = retrieval_date_for(doc, markdown, fallback_retrieval_date)
    chunks: list[Chunk] = []
    index = 0
    for heading, body in split_sections(strip_frontmatter(markdown)):
        packed = pack_units(heading, atomic_units(body))
        for text, protected in packed:
            if not text.strip():
                continue
            index += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}-{index:03d}",
                    doc_id=doc["doc_id"],
                    document_name=doc["document_name"],
                    publisher=doc["publisher"],
                    year=int(doc["year"]),
                    source_url=doc["source_url"],
                    retrieval_date=retrieval_date,
                    section_heading=heading,
                    text=text.strip(),
                    protected=protected,
                    char_count=len(text.strip()),
                )
            )
    return _merge_tiny_chunks(chunks)


def _merge_tiny_chunks(chunks: list[Chunk], min_chars: int = 40) -> list[Chunk]:
    """Fold leftover headings and one-line fragments into the next same-doc chunk."""
    if not chunks:
        return chunks
    merged: list[Chunk] = []
    carry: Chunk | None = None
    for chunk in chunks:
        if carry:
            if carry.doc_id == chunk.doc_id and not chunk.protected:
                text = f"{carry.text}\n\n{chunk.text}".strip()
                chunk.text = text
                chunk.char_count = len(text)
                if carry.section_heading != chunk.section_heading:
                    chunk.section_heading = f"{carry.section_heading}; {chunk.section_heading}"
            else:
                merged.append(carry)
            carry = None
        if not chunk.protected and chunk.char_count < min_chars:
            carry = chunk
            continue
        merged.append(chunk)
    if carry:
        merged.append(carry)
    for i, chunk in enumerate(merged, 1):
        chunk.chunk_id = f"{chunk.doc_id}-{i:03d}"
    return merged


def build_chunks() -> list[dict]:
    """Chunk the markdown files named in the manifest. Raw downloads are not read."""
    manifest = load_manifest()
    root_date = str(manifest.get("retrieval_date") or "")
    all_chunks: list[Chunk] = []
    for doc in manifest["documents"]:
        path = EXTRACTED_DIR / doc["extracted_file"]
        markdown = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_document(doc, markdown, fallback_retrieval_date=root_date))
    return [asdict(chunk) for chunk in all_chunks]

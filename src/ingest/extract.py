"""Extract written official guidance from raw HTML and PDF files.

Only project-required content is kept: heading-structured prose, intact tables,
and numbered recommendations. Site chrome, related-news blocks, reference lists,
methods/GRADE pages, and personal calorie-target tables are dropped.

Each extracted markdown file starts with YAML frontmatter so publisher, year,
source URL and retrieval date sit on the document itself. The chunker strips
that frontmatter before splitting.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup, Tag

from src.config import EXTRACTED_DIR, ROOT
from src.ingest.chunk import load_manifest

RAW_DIR = ROOT / "data" / "raw"

# Filenames actually present in data/raw (fetch_corpus.py uses different names).
RAW_FILES = {
    "icmr_nin_my_plate": "icmr_nin_my_plate_2024.pdf",
    "sfa_reusing_cooking_oils": "sfa_reusing_oils.html",
    "who_healthy_diet": "who_healthy_diet.html",
    "who_sfa_tfa_guideline": "who_sfa_tfa_guideline_summary.txt",
    "eatwell_guide": "eatwell_quick_guide.pdf",
    "health_canada_recommendations": "context (1).htm",
    "foodsafety_cold_storage": "foodsafety_cold_storage.pdf",
}

CHROME_LINE_RE = re.compile(
    r"^(?:"
    r"\d{1,2}/\d{1,2}/\d{2,4}|"
    r"\d+/\d+$|"
    r"Home\s*»|"
    r"An official website|"
    r"Here’s how you know|"
    r"Return to top|"
    r"Donate$|"
    r"A-\s*$|"
    r"A\+$|"
    r"POLICY BRIEF$|"
    r"PHE publications$|"
    r"PHE supports the UN$|"
    r"gateway number:|"
    r"https?://|"
    r"Canada\.ca$|"
    r"Departments and agencies$"
    r")",
    re.I,
)
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
CITATION_MARK_RE = re.compile(r"(?<!\d)([0-9]{1,3}(?:,\s*[0-9]{1,3})*)(?=\s|[.,;:)])")


@dataclass
class ExtractedDocument:
    doc_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    retrieval_date: str
    raw_file: str
    extracted_file: str
    markdown: str
    kept: str
    dropped: str


def extract_all() -> list[ExtractedDocument]:
    manifest = load_manifest()
    retrieval_date = manifest["retrieval_date"]
    results: list[ExtractedDocument] = []
    for doc in manifest["documents"]:
        results.append(extract_document(doc, retrieval_date))
    return results


def extract_document(doc: dict, retrieval_date: str) -> ExtractedDocument:
    raw_name = RAW_FILES[doc["doc_id"]]
    raw_path = RAW_DIR / raw_name
    body, kept, dropped = _DISPATCH[doc["doc_id"]](raw_path)
    markdown = _with_frontmatter(doc, retrieval_date, body)
    return ExtractedDocument(
        doc_id=doc["doc_id"],
        document_name=doc["document_name"],
        publisher=doc["publisher"],
        year=int(doc["year"]),
        source_url=doc["source_url"],
        retrieval_date=retrieval_date,
        raw_file=raw_name,
        extracted_file=doc["extracted_file"],
        markdown=markdown,
        kept=kept,
        dropped=dropped,
    )


def write_extracted(documents: list[ExtractedDocument]) -> Path:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for item in documents:
        path = EXTRACTED_DIR / item.extracted_file
        path.write_text(item.markdown.rstrip() + "\n", encoding="utf-8")
        record = asdict(item)
        record.pop("markdown")
        index.append(record)
    index_path = EXTRACTED_DIR / "_index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return index_path


def _with_frontmatter(doc: dict, retrieval_date: str, body: str) -> str:
    lines = [
        "---",
        f"doc_id: {doc['doc_id']}",
        f"document_name: {json.dumps(doc['document_name'], ensure_ascii=False)}",
        f"publisher: {json.dumps(doc['publisher'], ensure_ascii=False)}",
        f"year: {int(doc['year'])}",
        f"source_url: {doc['source_url']}",
        f"retrieval_date: {retrieval_date}",
        "---",
        "",
        body.strip(),
        "",
    ]
    return "\n".join(lines)


def _collapse(text: str) -> str:
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = text.replace("\u00a0", " ").replace("\u2011", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


RUNNING_HEADER_RE = re.compile(
    r"^(?:"
    r"A Quick Guide to the Government.?s Healthy Eating Recommendations|"
    r"POLICY BRIEF|"
    r"Guideline summary|"
    r"Remarks for all (?:SFA|TFA) recommendations|"
    r"1 WHO guidance on polyunsaturated fatty acids is currently being updated\.?"
    r")$",
    re.I,
)


def normalize_prose(text: str) -> str:
    """Join PDF soft wraps into paragraphs; keep bullets and blank-line breaks."""
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = text.replace("\u00a0", " ").replace("\u2011", "-")
    paragraphs: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            paragraphs.append(" ".join(buf))
            buf.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if CHROME_LINE_RE.match(line) or RUNNING_HEADER_RE.match(line):
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        if line.startswith(("- ", "# ", "## ", "### ", "| ", "o ")):
            flush()
            paragraphs.append(line)
            continue
        if re.match(r"^\d+\.\s+WHO ", line):
            flush()
            buf.append(line)
            continue
        if paragraphs and paragraphs[-1].startswith("o ") and not buf:
            paragraphs[-1] = paragraphs[-1] + " " + line
            continue
        if buf and re.match(r"^\d+\.\s+WHO ", buf[0]):
            buf.append(line)
            continue
        buf.append(line)
    flush()
    return "\n\n".join(paragraphs).strip()


OCR_FIXES = (
    ("Afetr", "After"),
    ("afetr", "after"),
    ("Lefotvers", "Leftovers"),
    ("chifof n", "chiffon"),
    ("1 day .", "1 day"),
)


def _fix_ocr(text: str) -> str:
    for src, dest in OCR_FIXES:
        text = text.replace(src, dest)
    return text


def _foodsafety_intro(lines: list[str]) -> str:
    text = _collapse(" ".join(lines))
    text = re.sub(r"\s+beverages\.?$", "", text)
    cut = text.find("indefinitely.")
    if cut != -1:
        text = text[: cut + len("indefinitely.")]
    return text


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(value: str) -> str:
        return " ".join((value or "").replace("\n", " ").split())

    headers = [cell(h) for h in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = list(row) + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(cell(c) for c in padded[: len(headers)]) + " |")
    return "\n".join(lines)


def _html_to_markdown(root: Tag, stop_headings: set[str] | None = None) -> str:
    stop = {h.lower() for h in (stop_headings or set())}
    parts: list[str] = []
    parent_h2 = ""

    def emit(block: str) -> None:
        block = _collapse(block)
        if block:
            parts.append(block)

    for el in root.descendants:
        if not isinstance(el, Tag):
            continue
        if el.name in {"script", "style", "nav", "footer", "aside"}:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            heading = _collapse(el.get_text(" ", strip=True))
            if heading.lower() in stop:
                break
            if el.name == "h2":
                parent_h2 = heading
            if el.name == "h3" and parent_h2.lower().startswith("who guidance"):
                emit(f"## {parent_h2}: {heading}")
            elif el.name == "h1":
                emit(f"# {heading}")
            elif el.name == "h2" and heading.lower().startswith("who guidance"):
                continue
            elif el.name == "h2":
                emit(f"## {heading}")
            else:
                emit(f"### {heading}")
            continue
        if el.name == "p" and el.parent is not None and el.parent.name not in {"li"}:
            text = _collapse(el.get_text(" ", strip=True))
            if text:
                emit(text)
            continue
        if el.name in {"ul", "ol"} and not el.find_parent(["ul", "ol"]):
            items = []
            for li in el.find_all("li", recursive=False):
                item = _collapse(li.get_text(" ", strip=True))
                if item:
                    items.append(f"- {item}")
            if items:
                emit("\n".join(items))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Document extractors
# ---------------------------------------------------------------------------


def _extract_sfa(path: Path) -> tuple[str, str, str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    blocks = soup.select("div.sfContentBlock")
    body_blocks: list[Tag] = []
    for block in blocks:
        heading = block.find(["h1", "h2", "h3"])
        title = heading.get_text(" ", strip=True).lower() if heading else ""
        if title.startswith("reference"):
            break
        body_blocks.append(block)
    root = BeautifulSoup("<div></div>", "lxml").div
    for block in body_blocks:
        root.append(block)
    body = _html_to_markdown(root)
    if not body.startswith("# "):
        body = "# Reusing Cooking Oils\n\n" + body
    return body, "guidance sections through consumer tips", "navigation, references, related publications"


def _extract_who_healthy_diet(path: Path) -> tuple[str, str, str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    article = soup.select_one("article.sf-detail-body-wrapper") or soup.select_one("article")
    if article is None:
        raise ValueError(f"WHO fact sheet article not found in {path}")
    body = _html_to_markdown(article, stop_headings={"who response", "resources"})
    if not body.startswith("# "):
        body = "# Healthy diet\n\n" + body
    return body, "key facts through how to promote healthy diets", "WHO response, resources, related news"


def _extract_icmr(path: Path) -> tuple[str, str, str]:
    doc = pymupdf.open(path)
    title = (
        'Promotion of "My Plate for the Day" and physical activity among the '
        "population to prevent all forms of malnutrition and NCDs in the country"
    )
    problem, need = _icmr_problem_statement(doc[0])
    table_md, notes = _icmr_plate_table(doc[1])
    later = _icmr_later_sections(doc)
    if len(need) > len(later["need"]):
        later["need"] = need
    parts = [
        f"# {title}",
        "## Problem statement and summary",
        problem,
        "## Need for the development of My Plate concept",
        later["need"],
        "## My Plate for the Day food groups",
        "The following table is the official My Plate for the Day composition for a 2000 Kcal reference plate.",
        table_md,
        "Notes attached to the plate:",
        notes,
        "## Recommendations",
        later["recommendations"],
        "## Interventions",
        later["interventions"],
        "## Advantages of My Plate",
        later["advantages"],
        "## Outcomes",
        later["outcomes"],
    ]
    return "\n\n".join(p for p in parts if p), "policy-brief prose and plate table", "author block, page chrome, suggested citation"


def _icmr_problem_statement(page: pymupdf.Page) -> str:
    columns: list[list[str]] = [[], [], []]
    need_parts: list[str] = []
    for block in page.get_text("blocks"):
        x0, y0, _x1, _y1, text, *_rest = block
        text = _collapse(str(text))
        if not text or text in {"POLICY BRIEF"} or re.fullmatch(r"\d+", text):
            continue
        if "Need for the development" in text:
            continue
        if y0 > 600 and x0 > 200:
            need_parts.append(text)
            continue
        if y0 < 175:
            continue
        if x0 < 200:
            columns[0].append(text)
        elif x0 < 390:
            columns[1].append(text)
        else:
            columns[2].append(text)
    prose = " ".join(" ".join(col) for col in columns if col)
    return normalize_prose(prose), normalize_prose(" ".join(need_parts))


_ICMR_VALUE_RE = re.compile(r"^(?:~\d+|\d+|~?\d+\s*%E|-)$")


def _icmr_plate_table(page: pymupdf.Page) -> tuple[str, str]:
    raw = page.get_text("text")
    if "Cereals" not in raw or "MY PLATE FOR THE DAY" not in raw:
        raise ValueError("ICMR plate table not found")
    block = raw[raw.find("Cereals") : raw.find("MY PLATE FOR THE DAY")]
    name_parts: list[str] = []
    values: list[str] = []
    rows: list[list[str]] = []

    def flush() -> None:
        nonlocal name_parts, values
        if name_parts and len(values) == 6:
            name = " ".join(name_parts).replace("/ ", "/")
            rows.append([name, *values])
        name_parts, values = [], []

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if _ICMR_VALUE_RE.match(line):
            values.append(line)
            if len(values) == 6:
                flush()
            continue
        if values:
            flush()
        name_parts.append(line)
    flush()
    if len(rows) < 8:
        raise ValueError(f"ICMR plate table parsed {len(rows)} rows, expected 8")
    headers = [
        "Food groups (2000 Kcal)",
        "Foods to be consumed raw weight (g/day)",
        "% of Energy from each food group/day",
        "Total E from each food group/day (Kcal)",
        "Total protein from each food group/day (g)",
        "Total fat from each food group/day (g)",
        "Total Carbs from each food group/day (g)",
    ]
    note_at = raw.find("Note:")
    notes = []
    if note_at != -1:
        for line in raw[note_at:].splitlines():
            line = line.strip()
            if not line or line in {"MY PLATE FOR THE DAY"}:
                continue
            line = re.sub(r"^(?:Note:\s*|\*\s*|\+\s*|#\s*|\$\s*)", "", line).strip()
            if line:
                if not line.endswith("."):
                    line += "."
                notes.append(f"- {line}")
    return _md_table(headers, rows), "\n".join(notes)


def _icmr_later_sections(doc: pymupdf.Document) -> dict[str, str]:
    text = "\n".join(page.get_text("text") for page in doc)
    text = text.replace("Communi-\ncation", "Communication").replace("communi- cation", "communication")
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    need = _between(text, "Need for the development of My Plate concept", "Food groups")
    recs = _icmr_bullets(_between(text, "RECOMMENDATIONS", "INTERVENTIONS"))
    inter = _icmr_bullets(_between(text, "INTERVENTIONS", "ADVANTAGES OF MY PLATE"))
    adv = normalize_prose(_between(text, "ADVANTAGES OF MY PLATE", "OUTCOMES"))
    outcome_raw = text.split("OUTCOMES", 1)[-1]
    outcome_raw = re.sub(
        r"SUGGESTED CITATION.*?Hyderabad\.",
        "",
        outcome_raw,
        flags=re.S | re.I,
    )
    return {
        "need": normalize_prose(need),
        "recommendations": recs,
        "interventions": inter,
        "advantages": adv,
        "outcomes": _icmr_bullets(outcome_raw),
    }


def _icmr_bullets(text: str) -> str:
    chunks = re.split(r"[•]", text)
    items = []
    for chunk in chunks:
        item = normalize_prose(chunk)
        if len(item) < 20:
            continue
        if item.lower().startswith("policy brief"):
            continue
        if item.lower().startswith("r. hemalatha"):
            continue
        item = re.sub(r"^o\s+", "  - ", item, flags=re.M)
        items.append(f"- {item}" if not item.startswith("  - ") and not item.startswith("- ") else item)
    return "\n".join(items)


def _between(text: str, start: str, end: str) -> str:
    pattern = re.compile(re.escape(start) + r"(.*?)" + re.escape(end), re.S | re.I)
    match = pattern.search(text)
    return match.group(1) if match else ""


def _slice_from(text: str, start: str, end: str) -> str:
    start_at = text.find(start)
    end_at = text.find(end, start_at + 1) if start_at != -1 else -1
    if start_at == -1 or end_at == -1:
        return ""
    return text[start_at:end_at]


def _extract_eatwell(path: Path) -> tuple[str, str, str]:
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text("text") for page in doc)
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    body = ["# The Eatwell Guide"]
    body.append("## Executive summary")
    body.append(_eatwell_exec(text))
    body.append("## The Eatwell Guide")
    body.append(
        "The Eatwell Guide is a pictorial representation of government healthy eating advice "
        "showing the proportions in which different types of foods are needed to have a well-balanced "
        "and healthy diet. The proportions shown are representative of your food consumption over the "
        "period of a day or even a week, not necessarily each meal time."
    )
    for heading, stop in [
        ("Fruit and vegetables", "Potatoes, bread, rice, pasta"),
        ("Potatoes, bread, rice, pasta and other starchy carbohydrates", "Dairy and alternatives"),
        ("Dairy and alternatives", "Beans, pulses, fish, eggs, meat and other proteins"),
        ("Beans, pulses, fish, eggs, meat and other proteins", "Oil and spreads"),
        ("Oil and spreads", "Foods high in fat, salt and sugar"),
        ("Foods high in fat, salt and sugar", "Aim to drink 6-8 glasses"),
    ]:
        section = normalize_prose(_eatwell_bullets(_between(text, heading, stop)))
        body.append(f"## {heading}")
        body.append(section)
    fluids = normalize_prose(_eatwell_bullets(_between(text, "Aim to drink 6-8 glasses of fluid every day", "Food labelling")))
    body.append("## Fluids")
    body.append("Aim to drink 6-8 glasses of fluid every day.\n\n" + fluids)
    labelling = normalize_prose(_eatwell_bullets(_between(text, "Food labelling", "Daily energy requirements")))
    body.append("## Food labelling")
    body.append(labelling)
    body.append("## Supplements")
    body.append(
        "Most of us can get all the nutrients we need by consuming a healthy, balanced and varied diet. "
        "However, there are certain groups of the population for whom certain dietary supplements are recommended."
    )
    body.append("### Folic acid")
    body.append(normalize_prose(_between(text, "Folic acid", "Vitamin D")))
    body.append("### Vitamin D")
    body.append(_eatwell_vitamin_d(text))
    body.append("### Vitamins A and C")
    body.append(normalize_prose(_between(text, "Vitamins A and C", "4. Further information")))
    return (
        "\n\n".join(p for p in body if p),
        "Eatwell food-group guidance and supplement advice",
        "cover, PHE about page, contents, artwork plate, daily energy/calorie targets, further information, references",
    )


def _eatwell_exec(text: str) -> str:
    chunk = _between(text, "1. Executive summary", "2. The Eatwell Guide")
    paras = []
    for piece in re.split(r"\n\s*1\.\d+\.\s*", chunk):
        item = normalize_prose(re.sub(r"(?:\)(\d+(?:,\d+)*)|(?<=[A-Za-z])\d+(?:,\d+)*)", lambda m: ")" if m.group(1) else "", piece))
        if item:
            paras.append(item)
    return "\n\n".join(paras)


def _eatwell_vitamin_d(text: str) -> str:
    """Reassemble vitamin D guidance; the PDF drops a footnote into the middle of the sentence."""
    start = text.find("It is recommended")
    end = text.find("Vitamins A and C")
    if start == -1 or end == -1 or "10µg" not in text[start:end]:
        raise ValueError("Vitamin D guidance missing from Eatwell PDF")
    chunk = text[start:end]
    foot_match = re.search(r"ii This is due to the fact.*?(?:intake\.)", chunk, flags=re.S)
    footnote = ""
    if foot_match:
        footnote = foot_match.group(0)
        chunk = chunk[: foot_match.start()] + "\n" + chunk[foot_match.end() :]
    chunk = re.sub(r"A Quick Guide to the Government.?s Healthy Eating Recommendations", "", chunk)
    chunk = re.sub(r"(?m)^\s*\d{1,2}\s*$", "", chunk)
    chunk = re.sub(r"(?<=[A-Za-z])(?:ii|\d+)(?=[.\s])", "", chunk)
    chunk = re.sub(r"\s*[•]\s*", "\n- ", chunk)
    joined: list[str] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            joined.append("")
            continue
        if joined and joined[-1].startswith("- ") and not line.startswith("- "):
            joined[-1] = f"{joined[-1]} {line}"
            continue
        joined.append(line)
    body = normalize_prose("\n".join(joined))
    body = body.replace("these groups\n\nshould take", "these groups should take")
    if footnote:
        footnote = normalize_prose(re.sub(r"^ii\s+", "", footnote))
        body = body + "\n\n" + footnote
    return body


def _eatwell_bullets(text: str) -> str:
    text = re.sub(r"\s*[•]\s*", "\n- ", text)
    text = re.sub(r"\n-\s+", "\n- ", text)
    return _collapse(text)


# Print-to-PDF column edges for the FoodSafety.gov storage chart.
_FS_COLS = ((40, 155), (155, 304), (304, 420), (420, 560))


def _fs_column_rules(page: pymupdf.Page) -> list[list[float]]:
    segments: list[tuple[float, float, float]] = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "re":
                continue
            rect = item[1]
            if rect.height > 2 or rect.width < 40:
                continue
            segments.append((rect.x0, rect.x1, rect.y0))
    per_column: list[list[float]] = []
    for x0, x1 in _FS_COLS:
        ys: list[float] = []
        for left, right, y in segments:
            if min(right, x1) - max(left, x0) > 40:
                ys.append(y)
        ys.sort()
        merged: list[float] = []
        for y in ys:
            if not merged or abs(y - merged[-1]) > 2.5:
                merged.append(y)
        per_column.append(merged)
    # Close rowspans at the last type-column rule. Lines below that are page footers.
    end = per_column[1][-1] if per_column[1] else 0
    for index, column in enumerate(per_column):
        if index == 1 or not column:
            continue
        trimmed = [y for y in column if y <= end + 1]
        if trimmed and trimmed[-1] < end - 1:
            trimmed.append(end)
        per_column[index] = trimmed
    return per_column


def _fs_join_words(words: list[tuple]) -> str:
    lines: dict[int, list[tuple]] = {}
    for word in words:
        lines.setdefault(round(word[1]), []).append(word)
    text = ""
    for y in sorted(lines):
        line = " ".join(word[4] for word in sorted(lines[y], key=lambda item: item[0]))
        if not text:
            text = line
        elif text.endswith("-"):
            text += line
        else:
            text += " " + line
    text = re.sub(r"\s+\.", ".", text)
    text = text.replace("trout.)", "trout)")
    text = re.sub(r" +", " ", text).strip()
    text = re.sub(r"^(\d day)\.$", r"\1", text)
    return _fix_ocr(text)


def _fs_chart_rows(page: pymupdf.Page) -> list[list[str]]:
    rules = _fs_column_rules(page)
    if len(rules[1]) < 2:
        return []
    words = [word for word in page.get_text("words") if 20 < word[1] < 810]

    def text_in(col: int, y0: float, y1: float) -> str:
        x0, x1 = _FS_COLS[col]
        picked = [
            word
            for word in words
            if y0 - 1 <= word[1] < y1 - 0.5 and x0 - 3 <= word[0] < x1 - 1
        ]
        return _fs_join_words(picked)

    def covering(col: int, mid: float) -> str:
        ys = rules[col]
        for start, stop in zip(ys, ys[1:]):
            if start - 1 <= mid < stop - 0.5:
                return text_in(col, start, stop)
        return ""

    rows: list[list[str]] = []
    for start, stop in zip(rules[1], rules[1][1:]):
        if stop - start < 8:
            continue
        mid = (start + stop) / 2
        row = [covering(0, mid), text_in(1, start, stop), covering(2, mid), covering(3, mid)]
        blob = " ".join(row)
        if row[1] in {"", "Type"} or row[2].startswith("Refrigerator"):
            continue
        if any(
            token in blob
            for token in ("Cold Food Storage", "foodsafety.gov", "Download Cold", "Home »", "Date Last", "Return to top")
        ):
            continue
        rows.append(row)
    return rows


def _extract_foodsafety(path: Path) -> tuple[str, str, str]:
    doc = pymupdf.open(path)
    intro_clip = doc[0].get_text("text", clip=pymupdf.Rect(0, 150, 600, 390))
    intro_lines = []
    for line in intro_clip.splitlines():
        line = line.strip()
        if not line or line == "Cold Food Storage Chart":
            continue
        if line.startswith("Looking for") or "FoodKeeper" in line or line.startswith("Download"):
            break
        intro_lines.append(line)
    rows: list[list[str]] = []
    last_food = ""
    for page in doc:
        for food, kind, fridge, freezer in _fs_chart_rows(page):
            if food:
                last_food = food
            elif last_food:
                food = last_food
            rows.append([food, kind, fridge, freezer])
    if len(rows) < 40:
        raise ValueError(f"FoodSafety chart parsed {len(rows)} rows; the grid looks incomplete")
    headers = [
        "Food",
        "Type",
        "Refrigerator [40°F (4°C) or below]",
        "Freezer [0°F (-18°C) or below]",
    ]
    grouped: dict[str, list[list[str]]] = {}
    order: list[str] = []
    for row in rows:
        food = row[0] or "Storage"
        if food not in grouped:
            grouped[food] = []
            order.append(food)
        grouped[food].append(row)
    parts = [
        "# Cold Food Storage Charts",
        _foodsafety_intro(intro_lines),
        "Refrigerator temperatures in this chart are 40°F (4°C) or below. Freezer temperatures are 0°F (-18°C) or below.",
    ]
    for food in order:
        parts.append(f"## {food}")
        parts.append(_md_table(headers, grouped[food]))
    return (
        "\n\n".join(parts),
        "intro plus complete per-category storage tables",
        "site chrome, FoodKeeper promo, download link, page footers",
    )


def _extract_who_sfa_tfa(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\u0000", "")
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    note_at = text.find("This document is a summary of a WHO guideline")
    note_end = text.find("Guideline summary", note_at + 10) if note_at != -1 else -1
    if note_at == -1 or note_end == -1:
        raise ValueError("WHO guideline summary note not found")
    note = normalize_prose(text[note_at:note_end])
    bg_start = text.find("Noncommunicable diseases (NCDs) are the")
    bg_end = text.find("Objective, scope and methods", bg_start + 1) if bg_start != -1 else -1
    background = ""
    if bg_start != -1 and bg_end != -1 and bg_end > bg_start:
        background = normalize_prose(_strip_citation_numbers(text[bg_start:bg_end]))
    cut = background.find("Although CVDs typically present later in life")
    if cut != -1:
        background = background[:cut].rstrip()
    obj_at = text.find("The objective of this guideline is to provide updated guidance")
    obj_end = text.find("The evidence", obj_at) if obj_at != -1 else -1
    if obj_at == -1 or obj_end == -1:
        raise ValueError("WHO guideline objective not found")
    objective_raw = text[obj_at:obj_end]
    objective_raw = re.sub(
        r"The guideline was developed by the WHO Nutrition Guidance Expert Advisory Group.*?public consultations\.\s*",
        "",
        objective_raw,
        flags=re.S,
    )
    footnote_at = re.search(r"\s+1\s+https?://", objective_raw)
    if footnote_at:
        objective_raw = objective_raw[: footnote_at.start()]
    objective = normalize_prose(_strip_citation_numbers(objective_raw))
    strength = normalize_prose(_between(text, "Box 1. Strength of WHO recommendations", "The reasoning behind"))
    strength = re.sub(r"\s+as noted elsewhere in this summary\.?", ".", strength)
    sfa_recs = _who_numbered(
        _slice_from(text, "1. WHO recommends that adults and children reduce saturated fatty acid", "Rationale and remarks")
    )
    sfa_remarks = _who_remarks(_between(text, "Remarks for Recommendation 3", "TFA recommendations"))
    tfa_recs = _who_numbered(
        _slice_from(text, "1. WHO recommends that adults and children reduce trans-fatty acid", "Rationale for TFA recommendations")
    )
    tfa_remarks = _who_remarks(_between(text, "Remarks for TFA recommendation 3", "Translation and implementation"))
    impl_start = text.find("The recommendations in this guideline should be considered in conjunction")
    impl_end = text.rfind("\nReferences")
    impl = ""
    if impl_start != -1 and impl_end != -1 and impl_end > impl_start:
        impl = text[impl_start:impl_end]
        impl = impl.replace("▶", "\n- ").replace("", "\n- ")
        impl = re.sub(r"\s+References\s*$", "", impl)
        impl = normalize_prose(_strip_citation_numbers(impl))
    body = "\n\n".join(
        [
            "# Saturated fatty acid and trans-fatty acid intake for adults and children: WHO guideline",
            "## Note on the source text",
            note,
            "## Background",
            background,
            "## Objective and scope",
            objective,
            "## Strength of WHO recommendations",
            strength,
            "## SFA recommendations",
            _who_recommendation_context(text),
            sfa_recs,
            "## Remarks for SFA recommendations",
            sfa_remarks,
            "## TFA recommendations",
            tfa_recs,
            "## Remarks for TFA recommendations",
            tfa_remarks,
            "## Translation and implementation",
            impl,
        ]
    )
    return body, "recommendations, remarks, implementation", "copyright front matter, methods, GRADE evidence, rationale, reference list"


def _strip_citation_numbers(text: str) -> str:
    text = re.sub(r"\s*\(\d+(?:[–-]\d+)?(?:,\s*\d+(?:[–-]\d+)?)*\)", "", text)
    return _collapse(text)


def _who_numbered(text: str) -> str:
    rebuilt = []
    for match in re.finditer(r"\d+\.\s+WHO .*?(?=\n\d+\.\s+WHO |\Z)", text, flags=re.S):
        item = normalize_prose(match.group(0))
        if item.startswith(("1. WHO", "2. WHO", "3. WHO")):
            rebuilt.append(item)
    return "\n".join(rebuilt)


def _who_remarks(text: str) -> str:
    pieces = re.split(r"[▶•]", text)
    items = []
    for piece in pieces:
        item = normalize_prose(_strip_citation_numbers(piece))
        if item.lower().startswith("remarks"):
            continue
        if "Reynolds et al" in item or "systematic review of prospective observational studies" in item:
            continue
        if len(item) < 40:
            continue
        items.append(f"- {item}")
    return "\n".join(items)


def _who_recommendation_context(text: str) -> str:
    start = text.find("All recommendations for SFA and TFA should be considered")
    end = text.find("SFA recommendations", start) if start != -1 else -1
    if start == -1 or end == -1:
        raise ValueError("WHO recommendation context sentence not found")
    chunk = text[start:end]
    chunk = re.sub(r"An explanation of the strength of WHO recommendations.*", "", chunk, flags=re.S)
    chunk = re.sub(r"\n\s*1\s*\n", " ", chunk)
    context = normalize_prose(_strip_citation_numbers(chunk))
    footnote = "WHO guidance on polyunsaturated fatty acids is currently being updated."
    if footnote in text:
        context = context + " " + footnote
    return context


def _inline_text(el: Tag) -> str:
    parts: list[str] = []
    for child in el.children:
        name = getattr(child, "name", None)
        if name in {"ul", "ol"}:
            continue
        if name is None:
            parts.append(str(child))
        else:
            parts.append(child.get_text(" ", strip=True))
    return _collapse(" ".join(parts))


def _recommendation_sections(ul: Tag) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for child in ul.children:
        name = getattr(child, "name", None)
        if name == "li":
            title = _inline_text(child).strip().rstrip(".").strip()
            bullets: list[str] = []
            nested = child.find("ul")
            if isinstance(nested, Tag):
                bullets.extend(_inline_text(item) for item in nested.find_all("li", recursive=False))
            current = (title, bullets)
            sections.append(current)
        elif name == "ul" and current is not None:
            title, bullets = current
            for item in child.find_all("li", recursive=False):
                bullets.append(_inline_text(item))
            sections[-1] = (title, bullets)
    return [(title, bullets) for title, bullets in sections if title]


def _extract_health_canada(path: Path) -> tuple[str, str, str]:
    """Parse the saved canada.ca recommendations page. The PDF is the same print."""
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one("main")
    if main is None or "Healthy eating recommendations" not in main.get_text(" ", strip=True):
        raise ValueError(f"Healthy eating recommendations page not found in {path}")
    for node in main.select(".well, .pagedetails, .visible-print, nav, header"):
        node.decompose()
    intro = ""
    habit = ""
    for paragraph in main.find_all("p"):
        text = _collapse(paragraph.get_text(" ", strip=True))
        if text.lower().startswith("healthy eating is more than"):
            intro = text
        elif text.lower().startswith("make it a habit"):
            habit = text.strip().rstrip(".").strip()
    top_lists = [ul for ul in main.find_all("ul") if ul.parent is not None and ul.parent.name not in {"ul", "li"}]
    if len(top_lists) < 2:
        raise ValueError("Health Canada recommendation lists not found")
    parts = ["# Healthy eating recommendations", intro]
    for title, bullets in _recommendation_sections(top_lists[0]):
        parts.append(f"## {title}")
        if bullets:
            parts.append("\n".join(f"- {item}" for item in bullets if item))
    if habit:
        parts.append(f"## {habit}")
    for title, bullets in _recommendation_sections(top_lists[1]):
        parts.append(f"## {title}")
        if bullets:
            parts.append("\n".join(f"- {item}" for item in bullets if item))
    return (
        "\n\n".join(part for part in parts if part),
        "2019 recommendation list from the saved canada.ca page",
        "site chrome, download link, publication sidebar, recipe footer; the PDF is the same page print",
    )


_DISPATCH = {
    "icmr_nin_my_plate": _extract_icmr,
    "sfa_reusing_cooking_oils": _extract_sfa,
    "who_healthy_diet": _extract_who_healthy_diet,
    "who_sfa_tfa_guideline": _extract_who_sfa_tfa,
    "eatwell_guide": _extract_eatwell,
    "health_canada_recommendations": _extract_health_canada,
    "foodsafety_cold_storage": _extract_foodsafety,
}

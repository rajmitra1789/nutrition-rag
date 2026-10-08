"""Parse data/raw into heading-structured markdown in data/processed/extracted."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import EXTRACTED_DIR
from src.ingest.extract import extract_all, write_extracted


LEFTOVER_DUMPS = (
    "eatwell_quick_guide.txt",
    "icmr_nin_my_plate_2024.txt",
)


def main() -> None:
    documents = extract_all()
    write_extracted(documents)
    for name in LEFTOVER_DUMPS:
        path = EXTRACTED_DIR / name
        if path.exists():
            path.unlink()
            print(f"removed leftover dump {path}")
    for item in documents:
        chars = len(item.markdown)
        print(f"{item.doc_id}: {chars} chars | kept {item.kept} | dropped {item.dropped}")
    print(f"Wrote {len(documents)} documents to {EXTRACTED_DIR}")


if __name__ == "__main__":
    main()

"""Download source files that are publicly reachable and record provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MANIFEST_PATH
from src.ingest.chunk import load_manifest

RAW = ROOT / "data" / "raw"
USER_AGENT = (
    "NutritionGuidanceRAG/1.0 (+https://github.com; retrieval for public dietary guidance)"
)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    provenance = {
        "retrieval_date": manifest["retrieval_date"],
        "downloads": [],
    }
    with httpx.Client(follow_redirects=True, timeout=60.0, headers={"User-Agent": USER_AGENT}) as client:
        for doc in manifest["documents"]:
            url = doc["source_url"]
            suffix = ".pdf" if doc["format"] == "pdf" else ".html"
            dest = RAW / f"{doc['doc_id']}{suffix}"
            try:
                response = client.get(url)
                response.raise_for_status()
                dest.write_bytes(response.content)
                status = "ok"
                size = dest.stat().st_size
            except Exception as exc:
                status = f"failed: {exc}"
                size = 0
            provenance["downloads"].append(
                {
                    "doc_id": doc["doc_id"],
                    "url": url,
                    "path": str(dest.relative_to(ROOT)),
                    "status": status,
                    "bytes": size,
                    "retrieval_date": manifest["retrieval_date"],
                }
            )
            print(f"{doc['doc_id']}: {status}")
    (RAW / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

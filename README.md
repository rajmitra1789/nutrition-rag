# Dietary Guidance RAG Chatbot

A Streamlit assistant that answers questions about food, nutrition, and food safety **only** from seven official public guidance documents. Every claim is cited. If the corpus does not cover the question, it says so and names what was searched.

This is not a medical service. It will not diagnose, treat, prescribe, set calorie targets, or say what anyone should weigh.

## What it does

- Retrieves from a ChromaDB index of official guidance
- Can search **all documents** or **one named document**
- Answers **per source** when two or more documents cover the same topic (for example cooking oil). It never blends two publishers into one claim
- Enforces two refusals **in code**, not only in the prompt:
  1. **Not in corpus** — no retrieved chunk clears the similarity floor
  2. **Out of scope** — medical advice, personal calorie targets, or weight targets

Nutrient numbers for individual foods are out of scope (Milestone 3).

## Corpus

Retrieval date for every document: **2026-10-05**. Metadata lives in `src/corpus/manifest.json`.

| Document | Publisher | Year | Source |
| --- | --- | --- | --- |
| My Plate for the Day | ICMR-National Institute of Nutrition, India | 2024 | [PDF](https://www.nin.res.in/brief/Policy%20Brief%20My%20Plate%20J18%2024.pdf) |
| Reusing Cooking Oils | Singapore Food Agency | 2024 | [page](https://www.sfa.gov.sg/food-safety-tips/food-risk-concerns/risk-at-a-glance/reusing-cooking-oils) |
| Healthy diet | World Health Organization | 2026 | [fact sheet](https://www.who.int/news-room/fact-sheets/detail/healthy-diet) |
| Saturated fatty acid and trans-fatty acid intake for adults and children: WHO guideline | World Health Organization | 2023 | [publication](https://www.who.int/publications/i/item/9789240073630) |
| The Eatwell Guide | Office for Health Improvement and Disparities (OHID), GOV.UK | 2024 | [publication](https://www.gov.uk/government/publications/the-eatwell-guide) |
| Healthy eating recommendations | Health Canada | 2019 | [page](https://www.canada.ca/en/health-canada/services/food-guide/explore/healthy-eating-recommendations.html) |
| Cold Food Storage Charts | FoodSafety.gov (U.S. government) | 2023 | [charts](https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts) |

Written prose only. Visual plates and site chrome are not ingested.

**Source notes**

- **WHO SFA/TFA:** the official 24-page guideline summary (ISBN 978-92-4-008359-2). WHO states it is extracted directly from the full 134-page guideline and is identical to that content. Methods pages and the reference list are omitted so retrieval stays on recommendations and remarks.
- **Eatwell Guide:** the written companion *A Quick Guide to the Government’s Healthy Eating Recommendations*, hosted on the GOV.UK Eatwell publication page last updated 2 January 2024. The one-page artwork PDF is not used.
- **Health Canada:** the 2019 recommendation list (Pub. 180394) from the saved canada.ca page. Supporting pages are not in `data/raw` and are not ingested.

## Chunking — what was chosen and what it cost

The chunker in `src/ingest/chunk.py` splits on markdown headings, then packs paragraphs toward a **1,600-character** target (hard cap 2,400 for ordinary prose).

**Protected units are never cut:**

- markdown tables (the ICMR plate table; each FoodSafety.gov food-category table)
- consecutive numbered recommendation lists (WHO SFA and TFA recommendations stay as one block)

A leftover fragment under 40 characters is folded into the next same-document chunk so orphan dates and headings do not become their own vectors.

| Measure | Value |
| --- | --- |
| Chunks | 76 |
| Protected tables / numbered blocks | 16 |
| Median / mean characters | 573 / 806 |
| p95 / max characters | 2,391 / 4,301 |
| Median / mean estimated tokens | 135 / 176 |
| p95 / max estimated tokens | 475 / 893 |
| Chunks over 256 tokens | 18 |
| Chunks over 512 tokens | 3 |
| Chunks over 8,192 tokens | 0 |

Three unprotected paragraphs have no blank line, so they stay whole past the 2,400-character prose cap: WHO SFA/TFA Background (893 tokens), the ICMR problem statement (599), and the TFA remarks (524). Every protected table fits in 512 tokens.

**Cost of this choice:** more chunks than a coarser split (76 vs about 50), and a few short but complete recommendation sections. The gain is that no table row and no numbered WHO recommendation is split, which is what the brief requires.

Rebuild:

```bash
python scripts/build_chunks.py
python scripts/analyze_chunks.py
```

## Embedding — chosen after chunk analysis

`scripts/analyze_chunks.py` compared chunk length to common embedding windows.

| Model | Window | Decision |
| --- | --- | --- |
| `all-MiniLM-L6-v2` | 256 tokens | Rejected. Would truncate 18 chunks, including intact tables. |
| `BAAI/bge-small-en-v1.5` | 512 tokens | Rejected. Would truncate the three long paragraphs above. 33M parameters, 384 dimensions. |
| `snowflake-arctic-embed-m-long` | 2,048 tokens | Rejected. Window fits, but the ONNX file is 0.54 GB and queries need a prefix. |
| `nomic-embed-text-v1.5` | 8,192 tokens | Rejected. Window fits, but passages need a prefix and the ONNX file is 0.52 GB. |
| `jinaai/jina-embeddings-v2-small-en` | 8,192 tokens | **Chosen.** Covers every chunk (max 893 tokens). 33M parameters, 512 dimensions, 0.12 GB. |

**Serving cost:** the model is loaded with **FastEmbed (ONNX)**, not PyTorch. That keeps Streamlit Community Cloud under a workable memory budget. First boot downloads about 120 MB of ONNX weights.

Passages and queries are embedded as stored. This model does not use a query prefix.

Rebuild the index after changing chunks:

```bash
python scripts/build_index.py
```

## Retrieval and answers

- Store: persistent ChromaDB at `data/chroma`, cosine space
- Default `TOP_K=10`, `MAX_DOCS_IN_ANSWER=5`, similarity floor `MIN_SIMILARITY=0.75`
- Sidebar filter: all documents, or one `doc_id`
- Generation model: Groq `openai/gpt-oss-120b`, temperature 0.1
- Cross-document questions: one model call **per document**, then concatenated. The mixer never sees two publishers at once

Citations: each claim ends with a short tag `(Document name, Publisher, Year)`. Each document section header carries the full citation `(Document name, Publisher, Year, URL)`. When a document's passages do not answer, the model replies `NOT_COVERED` and that section is dropped. If every section is dropped, the not-in-corpus refusal names the documents searched.

## Run locally

Python 3.12+ (`runtime.txt` records 3.12, the version used on Streamlit Cloud). A local 3.14 venv also works with the ONNX embedder. `requirements.txt` holds exact runtime pins; `requirements-dev.txt` adds `pytest` and the ingest helpers.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY
python scripts/build_chunks.py
python scripts/build_index.py
streamlit run app.py
```

Get a Groq key at [console.groq.com](https://console.groq.com).

## Deploy on Streamlit Community Cloud

Full plan, checks, and troubleshooting: `docs/deployment-plan.md`.

1. Push this repo (include `data/processed/` and `data/chroma/`)
2. Create an app with entry point `app.py`; under **Advanced settings** choose Python 3.12
3. In **Advanced settings → Secrets** add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

The first visitor may wait while FastEmbed fetches `jinaai/jina-embeddings-v2-small-en` (about 120 MB).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

These assert the refusals and the “do not split tables or numbered recommendations” rule in code.

## Project layout

```
app.py
src/corpus/manifest.json
src/ingest/chunk.py
src/retrieve/index.py
src/answer/guardrails.py
src/answer/generate.py
data/processed/extracted/     # cleaned official prose
data/processed/chunks.json
data/chroma/                  # persistent index
scripts/build_chunks.py
scripts/analyze_chunks.py
scripts/build_index.py
docs/problemStatement.md
```

# Implementation plan

Phase order for building the dietary guidance chatbot in [problemStatement.md](problemStatement.md), in the shape recorded in [architecture.md](architecture.md).

Each phase has one exit. Later phases assume the previous exit is true. Do not start the answer layer until the index exists. The embedding model is the one Phase 5 records from the measured lengths in `data/processed/chunks.json`.

## What “done” means

A Streamlit chat app answers food, nutrition, and food safety questions from seven official guidance documents in a persistent ChromaDB collection. Generation is one Groq call per document (`openai/gpt-oss-120b`). Two refusals run in Python before or instead of generation: out of scope (regex) and not in corpus (similarity floor).

Nutrient numbers for individual foods stay out of scope.

## Phase map

| Phase | Builds | Requirement it closes |
| --- | --- | --- |
| 1 | Project skeleton and config | Stack |
| 2 | Corpus catalog | 5–7 authorities; publisher, year, URL, retrieval date |
| 3 | Written prose the index will read | Written prose only |
| 4 | Chunks with protected tables and lists | Chunk metadata; do not cut tables or numbered lists |
| 5 | Embedding choice from measured lengths | Embedding chosen after chunk analysis |
| 6 | Chroma collection | Vector index |
| 7 | Retrieve all documents, or one | Filter to one named document |
| 8 | Refusals in Python | Not in corpus; out of scope |
| 9 | Per-document answers and citations | Answer from chunks; cite every claim; never blend publishers |
| 10 | Streamlit chat | Deployable UI |
| 11 | Tests and the README cost note | Chunking cost is stated; refusals and filters are checked |
| 12 | Streamlit Community Cloud | Deployment |

## Phase 1 — Skeleton and configuration

**Goal.** A Python package whose defaults match the running system, with secrets kept out of source.

**Build**

- Package layout: `src/config.py`, `src/ingest/`, `src/retrieve/`, `src/answer/`, `app.py`, `scripts/`.
- `src/config.py` owns paths and defaults:

| Setting | Value |
| --- | --- |
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-small-en` (confirmed in Phase 5) |
| `EMBEDDING_QUERY_PREFIX` | `""` (Jina v2 uses no query or passage prefix) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` |
| `GROQ_MAX_TOKENS` | `320` |
| `GROQ_REASONING_EFFORT` | `low` |
| `PASSAGE_SCORE_GAP` | `0.01` |
| `PASSAGE_TOKEN_BUDGET` | `450` estimated tokens |
| `MIN_SIMILARITY` | `0.75` |
| `TOP_K` | `10` (final passage budget, not the candidate pool) |
| `PER_DOC_CHUNK_CAP` | `3` |
| `PER_DOC_WINDOW` | `0.04` |
| `MAX_DOCS_IN_ANSWER` | `5` |
| `TARGET_CHUNK_CHARS` | `1600` |
| `MAX_CHUNK_CHARS` | `2400` |
| `COLLECTION_NAME` | `dietary_guidance` |

- `GROQ_API_KEY` comes from the environment, then Streamlit secrets. No key in source.
- `.env.example` and `.streamlit/secrets.toml.example` document the key and nothing else. Tuning values live only in `src/config.py`, so local runs and the cloud use the same defaults. `.gitignore` ignores `.env` and `.streamlit/secrets.toml`.
- `requirements.txt`: `streamlit`, `chromadb`, `fastembed`, `groq`, `python-dotenv`, `pytest`.
- `runtime.txt`: `python-3.11.11`.

**Exit.** `src.config` imports, defaults match the table, and a missing key does not crash import.

## Phase 2 — Corpus catalog

**Goal.** Seven public guidance documents, each with publisher, year, source URL, and retrieval date.

**Build** `src/corpus/manifest.json`.

- Root `retrieval_date`: `2026-10-05`. Copy that date onto **each** document object as well. The brief asks for it on every document; a single root field is not enough.
- Seven `doc_id`s, in this order:

| `doc_id` | Publisher | Year | What is actually indexed |
| --- | --- | --- | --- |
| `icmr_nin_my_plate` | ICMR-National Institute of Nutrition, India | 2024 | The 4-page policy brief, as prose |
| `sfa_reusing_cooking_oils` | Singapore Food Agency | 2024 | The Reusing Cooking Oils page |
| `who_healthy_diet` | World Health Organization | 2026 | The Healthy diet fact sheet |
| `who_sfa_tfa_guideline` | World Health Organization | 2023 | The official 24-page guideline **summary**. Methods and the reference list stay out |
| `eatwell_guide` | OHID, GOV.UK | 2024 | *A Quick Guide to the Government’s Healthy Eating Recommendations*. The one-page artwork plate is not ingested |
| `health_canada_recommendations` | Health Canada | 2019 | The 2019 recommendation list |
| `foodsafety_cold_storage` | FoodSafety.gov | 2023 | The cold-storage category charts, as tables |

- Each row also has `document_name`, `source_url`, `format`, `extracted_file`, and a `notes` field when the indexed text is a stated substitute (WHO summary, Eatwell quick guide).
- `source_url` values are the seven URLs in the brief.

**Exit.** The manifest lists 5–7 recognised authorities, and every document object has publisher, year, source URL, and retrieval date.

## Phase 3 — Written prose

**Goal.** Seven heading-structured markdown files. The chunker reads these files, not `data/raw/`.

**Build**

1. Optional download: `scripts/fetch_corpus.py` GETs each `source_url` into `data/raw/`. It does not parse and is not called by chunking or indexing. Several live downloads are error pages (reCAPTCHA, “Page Moved”, Access Denied). Do not index those files.
2. Produce `data/processed/extracted/{extracted_file}` for every manifest row:
   - Headings as `#`–`######`.
   - Tables kept as markdown tables.
   - Numbered recommendations kept as numbered lists.
   - Site chrome, related-news blocks, reference lists, and visual plates dropped.
   - Written prose only.
3. The chunker contract is the markdown filename in the manifest. A parser (`src/ingest/extract.py`, `scripts/extract_corpus.py`) may produce those files. If a source cannot be parsed cleanly, clean that file offline and still point `extracted_file` at the result. Broken raw captures never become chunks.

**Exit.** All seven `extracted_file` paths exist, are heading-structured prose, and contain no artwork and no error-page HTML.

## Phase 4 — Chunking

**Goal.** Chunks that carry document metadata and never split a table or a numbered recommendation.

**Build** `src/ingest/chunk.py`, driven by `scripts/build_chunks.py`.

1. Read only the markdown files named in the manifest.
2. `split_sections` walks heading lines. Text before the first heading is filed under `"Document"`.
3. `atomic_units` marks two kinds of unit as **protected**:
   - a markdown table (`|` rows)
   - a run of numbered items (`^\s*\d+\.\s+\S`)
   Other text is paragraphs. A bullet list after prose starts a new unprotected unit.
4. Pack unprotected units toward `TARGET_CHUNK_CHARS` (1600). Flush before every protected unit. Split ordinary prose over `MAX_CHUNK_CHARS` (2400) only on blank lines. Append protected units whole. Never pass them to the long-prose splitter.
5. Fold leftovers under 40 characters into the next same-document, non-protected chunk.
6. Each chunk stores: `chunk_id`, `doc_id`, `document_name`, `publisher`, `year`, `source_url`, `retrieval_date`, `section_heading`, `text`, `protected`, `char_count`.
7. Write `data/processed/chunks.json`.

`chunk_id` form: `{doc_id}-{nnn}` (for example `icmr_nin_my_plate-006`).

**Exit.** `chunks.json` exists. The ICMR plate table is one chunk. Each FoodSafety.gov food-category table is one chunk. WHO SFA items 1–3 are one chunk and TFA items 1–3 are one chunk. Every chunk has document name, publisher, year, retrieval date, and section heading. The current file has 76 chunks, 16 of them protected. More chunks than a naive ~50-chunk split is the accepted cost.

## Phase 5 — Choose the embedding model

**Goal.** Pick the embedder from the chunks on disk. The earlier guess (max about 439 tokens, 84 chunks, `BAAI/bge-small-en-v1.5`) does not match `chunks.json`.

**Measured** from `data/processed/chunks.json`. Tokens are `max(1, round(words / 0.75))`, the same heuristic `scripts/analyze_chunks.py` writes. This is not a model tokenizer.

| | Characters | Estimated tokens |
| --- | --- | --- |
| Chunks | 76, of which 16 are protected | |
| min | 34 | 8 |
| median | 573 | 135 |
| mean | 806 | 176 |
| p95 | 2,391 | 475 |
| max | 4,301 | 893 |
| Over 256 tokens | | 18 chunks |
| Over 512 tokens | | 3 chunks |
| Over 2,048 or 8,192 tokens | | 0 |

Per document: WHO Healthy diet 15, Eatwell 15, FoodSafety.gov 14, WHO SFA/TFA 12, ICMR 10, Health Canada 7, SFA oils 3.

The three chunks over 512 tokens are unprotected single paragraphs. They have no blank line, so the Phase 4 splitter cannot cut them:

| `chunk_id` | Section | Characters | Estimated tokens |
| --- | --- | --- | --- |
| `who_sfa_tfa_guideline-002` | Background | 4,301 | 893 |
| `icmr_nin_my_plate-001` | Problem statement and summary | 2,912 | 599 |
| `who_sfa_tfa_guideline-009` | Remarks for TFA recommendations | 2,618 | 524 |

Every protected unit fits in 512 tokens. The longest is the FoodSafety.gov Eggs table, at 411 tokens.

**Build** `scripts/analyze_chunks.py`. It reads `chunks.json` and writes `data/processed/chunk_analysis.json` with these counts and the decision below.

Candidates are models FastEmbed can serve as ONNX, so Streamlit Community Cloud does not load PyTorch.

| Model | Window | FastEmbed ONNX | Decision |
| --- | --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | 256 | 0.09 GB, 384-d | Reject. Truncates 18 chunks, including protected tables (Eggs 411, Ham 296, ICMR plate 269). |
| `BAAI/bge-small-en-v1.5` | 512 | 0.07 GB, 384-d, 33M | Reject. Would drop the tail of the three paragraphs above. The Background chunk would lose about 380 of 893 tokens. |
| `snowflake/snowflake-arctic-embed-m-long` | 2,048 | 0.54 GB, 768-d | Reject. The window fits (max 893), but the file is several times larger than necessary and queries need a `query:` prefix. |
| `nomic-ai/nomic-embed-text-v1.5` | 8,192 | 0.52 GB, 768-d | Reject. The window fits, but passages must be stored with a `search_document:` prefix and the weights are heavier. |
| `jinaai/jina-embeddings-v2-small-en` | 8,192 | 0.12 GB, 512-d, 33M | **Choose.** Every chunk fits. The longest uses about 11% of the window. Same parameter class as bge-small. No query or passage prefix. |

Set `EMBEDDING_MODEL` to `jinaai/jina-embeddings-v2-small-en`. Set `EMBEDDING_QUERY_PREFIX` to `""`. First boot downloads about 120 MB.

Do not carry `MIN_SIMILARITY` over from bge-small. These vectors are unit length, and query scores on this collection sit in a band from about 0.56 to 0.92. Phase 7 sets the floor at `0.75`.

**Exit.** `chunk_analysis.json` records the comparison and `jinaai/jina-embeddings-v2-small-en`. No chunk exceeds 8,192 tokens. The README states the 1,600-character target, 76 chunks, 16 protected units, the percentiles above, the rejected embedders, and the cost versus a coarser split (76 versus about 50).

## Phase 6 — Vector index

**Goal.** A persistent Chroma collection of one 512-dimensional embedding per chunk, from the Phase 5 model.

**Build** `src/retrieve/index.py` `build_index()`, driven by `scripts/build_index.py`.

- Client: `chromadb.PersistentClient(path=data/chroma)`.
- Collection: `dietary_guidance`, cosine space (`hnsw:space = cosine`).
- Embedding function: FastEmbed `TextEmbedding("jinaai/jina-embeddings-v2-small-en")`. 512 dimensions. Do not keep a BGE-only wrapper, and do not prefix text with the BGE instruction string.
- Delete the collection if it exists, then create and `add`. Rebuilds are full replacements. A model change is a full rebuild: do not leave 384-dimensional bge vectors in the same collection as 512-dimensional Jina vectors.
- Id: `chunk_id`. Document field: chunk `text`. One record per object in `chunks.json` (76).
- Metadata: `doc_id`, `document_name`, `publisher`, `year`, `source_url`, `retrieval_date`, `section_heading`, `protected`. `retrieval_date` is on every chunk and is stored.
- `EMBEDDING_QUERY_PREFIX` is empty. Passages and queries are embedded as stored. Jina v2 was not trained with a query prefix.
- `data/chroma/` is committed. Cloud deploy does not rebuild the index. First boot still downloads the ONNX weights (about 120 MB) if they are not cached.

**Exit.** The collection has 76 vectors, each 512-dimensional, and metadata includes publisher, year, source URL, and retrieval date.

## Phase 7 — Retrieval

**Goal.** Search the whole corpus, or one named document, and keep only chunks that are near the query in this index.

**What the index actually is.** Phase 6 stored 76 vectors from `jinaai/jina-embeddings-v2-small-en`. Each is 512-dimensional and L2-normalised, so cosine similarity is the dot product and `similarity = 1.0 - distance`. There is no query prefix.

The vectors share a narrow cone. Chunk-to-chunk cosine is at least 0.644 across documents (median 0.756) and at least 0.678 inside a document (median 0.827). Probe queries land in the same band:

| Probe | Best similarity | Nearest chunk |
| --- | --- | --- |
| Reuse cooking oil | 0.869 | All 3 SFA oil chunks (0.840–0.869) |
| Saturated fat for adults | 0.843 | WHO Healthy diet fats, then Eatwell oils (0.813) and the SFA recommendations (0.803) |
| Raw eggs in the fridge | 0.844 | FoodSafety.gov Eggs. The next storage table is 0.789 |
| Ham in the fridge | 0.906 | FoodSafety.gov Ham. The next table is 0.847 |
| Opened hot dogs | 0.850 | FoodSafety.gov Hot dogs. The next table is 0.791 |
| Water instead of sugary drinks | 0.920 | Health Canada “Make water your drink of choice” (34 characters) |
| Servings of vegetables and fruit | 0.848 | Eatwell fruit and vegetables, then two ICMR plate chunks (0.823, 0.822) |
| Protein in a chicken breast | 0.789 | WHO protein guidance. The corpus has no per-food nutrient table |
| Kimchi | 0.786 | Health Canada “Enjoy your food” |
| Calories to lose weight | 0.758 | A WHO fats chunk |
| Capital of France | 0.670 | An ICMR outcomes chunk |
| Reset a wifi password | 0.717 | A FoodSafety.gov Ham chunk |

A floor of 0.36 sits under every probe, including France and wifi, so it never triggers the not-in-corpus refusal. On the later check set, in-corpus tops ran from 0.827 to 0.894 and off-topic tops topped out at 0.722 (wifi). `MIN_SIMILARITY` is `0.75`, inside that gap. There is no cross-document margin: a second publisher stays if its own best chunk clears 0.75.

Document size also shapes the result. SFA oils has 3 chunks. Health Canada has 7. ICMR has 10. WHO SFA/TFA has 12. FoodSafety.gov has 14 table chunks. WHO Healthy diet and Eatwell have 15 each. A flat top-8 for “raw eggs” is eight storage tables. The per-document cap and window are what stop that, not a margin against the global best.

**Build** `retrieve(query, doc_id=None, top_k=TOP_K)` in `src/retrieve/index.py`.

1. Embed the query as typed. `EMBEDDING_QUERY_PREFIX` is empty. Do not add the BGE instruction string.
2. `doc_id is None`: no `where` filter. `doc_id` set: `where={"doc_id": doc_id}`.
3. Request every matching row. The collection has 76 rows, and a filtered document has as few as 3. `n_results` must not exceed the rows the filter matches, and it must not be `TOP_K`. Taking 8 candidates first hides a second publisher once one document fills the list. Include documents, metadatas, and distances. `similarity = 1.0 - distance`.
4. Drop hits below `MIN_SIMILARITY` (`0.75`).
5. Inside each remaining document, keep at most `PER_DOC_CHUNK_CAP` (3) chunks, and only chunks within `PER_DOC_WINDOW` (`0.04`) of that document’s best score. Apply this on an all-documents search and on a single-document search. Do not drop a document for trailing the global best.
6. Order the survivors by similarity and stop at `TOP_K` (10). At 8, “How much cooking oil per day?” lost `icmr_nin_my_plate-004` (the Fats & Oils 27 g/day row, 0.762), which ranks ninth.
7. `enough` is true only when at least one hit remains.
8. `searched_documents` is the full manifest catalog, or the single row for `doc_id`. The not-in-corpus refusal names this list.

The per-document window keeps a single storage table when the next table in that document is more than 0.04 behind (eggs, ham, hot dogs). It keeps up to three chunks from one document when they sit inside that window (the SFA oil page; a leftovers question can still bring neighbouring storage tables). Other documents stay in the list when their best chunk is at least 0.75.

**Limits to leave visible.** Similarity picks the nearest chunk. It does not check that the chunk states the asked fact. “Safe internal temperature for chicken” scores 0.857 on the poultry storage chart. “What food groups are on a healthy plate?” ranks ICMR Advantages (0.857) above the plate table. A leftovers question keeps Soups and stews and Salad with Leftovers, because those tables sit within 0.04 of 0.841. Do not drop short chunks to avoid that: the 34-character water line is the correct hit at 0.920.

**Exit.** The same function answers “all documents” and “this `doc_id` only”. Hits under 0.75 are absent. An eggs question does not return eight storage tables. An empty result still reports which documents were searched.

## Phase 8 — Refusals in code

**Goal.** Both refusals are Python branches. The prompt may repeat them. It must not be the only enforcement.

**Build** `src/answer/guardrails.py`, called from `answer_question` before generation.

**Out of scope.** `classify_scope` runs first. A match returns `OUT_OF_SCOPE_MESSAGE` and does not retrieve. Patterns cover medical advice and calorie or weight targets, including anything about what a person should weigh. The message declines and points to a qualified professional.

**Not in corpus.** After retrieval, if `not result.enough`, return `not_in_corpus_refusal(result.searched_documents)`.

- One document: name it as `{name} ({publisher}, {year})`.
- Several: “I searched the following documents:” and one bullet per document.
- Empty catalog: fall back to “the loaded dietary guidance corpus”.

**Exit.** A calorie, weight, or medical-pattern question never calls retrieve or Groq. A question whose best hit is under 0.75 returns the refusal and names what was searched. Oil, fat, and fridge questions are not refused by the scope regex.

Known limit to keep visible in tests: the regex list is fixed. Phrasings it does not match still reach retrieval. The system prompt also forbids medical advice, but that line is not the enforcement.

## Phase 9 — Answers and citations

**Goal.** Answers use only retrieved chunks. Two publishers never share a model call. Every factual claim carries a citation.

**Build** `src/answer/generate.py`.

`answer_question(query, doc_id=None)`:

1. `classify_scope`. On refusal, stop.
2. `retrieve`. On `not enough`, return the not-in-corpus refusal.
3. Otherwise `answer_from_retrieval`.

`answer_from_retrieval`:

1. Group hits by `doc_id`.
2. Sort groups by each group’s max similarity.
3. Keep at most `MAX_DOCS_IN_ANSWER` (5). ICMR is the fifth document on the cooking-oil-per-day question.
4. One Groq chat completion per remaining group. Temperature `0.1`. Model `GROQ_MODEL`. Every call sets `max_tokens` to `GROQ_MAX_TOKENS` (320) and `reasoning_effort` to `low`. The free plan for this model allows 30 requests per minute, 1,000 per day, 8,000 tokens per minute, and 200,000 tokens per day. Reasoning tokens count toward the completion cap and toward the 8,000-token minute limit, so the effort stays low and the cap stays small. 320 is the longest measured completion on “What does the guidance say about cooking oil?” (252 tokens, SFA section, 11 uncapped runs) plus about 25% headroom. At 256 with full URLs inline, the SFA answer stopped on `finish_reason=length` mid-citation. `scripts/token_report.py` prints per-call prompt, completion, and reasoning tokens and the finish reason.
5. The system prompt names that one document and says: first decide whether the passages directly answer the question; if they do not, reply with exactly `NOT_COVERED` and nothing else; never summarize passages that do not answer; if the question asks for an amount, number, duration or limit and the passages do not state that quantity for the exact thing asked about, reply exactly `NOT_COVERED` (other numbers in the passages do not count); use only these passages; no outside knowledge; no medical advice; no calorie or weight targets; at most four short sentences. Keep that prompt to those lines.
6. Two citation forms, both built from hit metadata. The model ends every claim with the short tag, and the passage block shows it once as `Cite as:`:

`(Document name, Publisher, Year)`

The full citation, with the URL, appears once in the section header built in code:

`(Document name, Publisher, Year, URL)`

Send only the passages that call needs. Always send that document’s best chunk. Send another chunk from the same document only when it is within `PASSAGE_SCORE_GAP` (0.01) of that best score and the passage text still fits in `PASSAGE_TOKEN_BUDGET` (450 estimated tokens, `words / 0.75`). On the cooking-oil question this keeps both SFA chunks (0.845 and 0.840, about 310 tokens together) and drops the WHO translation chunk (0.016 behind the best WHO hit, about 467 tokens). On “How much cooking oil per day?” it keeps ICMR `-002` (153) and the My Plate table `-004` (269, 0.003 behind); at 400 the table was cut.
7. Drop any section whose reply is empty or contains `NOT_COVERED` anywhere. The marker is never shown. Head each remaining section with `**{document_name}** — {publisher} ({year}) (Document name, Publisher, Year, URL)` and separate sections with `---`. If every section is dropped, `answer_question` returns the not-in-corpus refusal naming `result.searched_documents`.
8. On HTTP 429, wait for the `retry-after` header (otherwise 5 seconds, and never more than 60) and retry that call once. If the retry is also 429, stop and return `Try again in a minute.`

**Exit.** A cooking-oil question that hits both SFA and WHO returns two labelled sections. The request for one section contains only that document’s passages. A factual sentence in the prompt contract ends with the short tag, and each section header carries the full citation with the URL. No call on the cooking-oil question ends with `finish_reason=length`. A `NOT_COVERED` section never reaches the user, and all-`NOT_COVERED` returns the not-in-corpus refusal. Generation is never called for a refusal.

## Phase 10 — Streamlit app

**Goal.** The only UI is `app.py`.

**Build**

- Title and caption: official guidance only; not medical advice; no calorie or weight targets.
- Sidebar selectbox “Retrieve from”: `all` or one `doc_id`. Map `all` to `doc_id=None`.
- Corpus list with source URLs, the out-of-scope note, and a clear-chat button.
- Chat input calls `answer_question(prompt, doc_id=...)`.
- Assistant message, plus an expander of retrieved passages. Hide the expander when the turn was a refusal.
- Session state stores `messages` with optional `hits`.
- Do not expose `TOP_K`, `MIN_SIMILARITY`, or the model name in the UI.
- `.streamlit/config.toml`: light theme, `server.headless = true`.

**Exit.** Local `streamlit run app.py` answers an in-corpus question with passages, refuses an out-of-scope question with no passages, and a sidebar pick limits retrieval to that document.

## Phase 11 — Tests and the cost write-up

**Goal.** The brief’s mechanical rules fail in pytest when broken, and the README records the chunking tradeoff.

**Build**

| Test | Asserts |
| --- | --- |
| `tests/test_chunking.py` | A table stays one protected unit. A numbered list stays one protected unit. Section headings survive. `pack_units` does not split a protected block |
| `tests/test_guardrails.py` | Calorie, weight, and medical examples refuse. Oil, fat, and fridge questions pass. Not-in-corpus text names the searched document |
| `tests/test_retrieval_filter.py` | The manifest has 7 documents. Filtering the catalog by `doc_id` returns one row |

The retrieval-filter test may check the catalog list without opening Chroma. The filter that hits the index still lives only in `retrieve()`. Add a Chroma-backed case only if the test environment can build a tiny collection; do not mock a second retrieval implementation.

README section “Chunking — what was chosen and what it cost” must include the 1,600-character target, chunk count, protected count, length stats, the rejected embedders, and the 76-versus-~50 cost.

**Exit.** `pytest` passes. The README cost section matches `chunk_analysis.json`.

## Phase 12 — Deploy

**Goal.** The app runs on Streamlit Community Cloud from the committed index.

**Build**

1. Commit `data/processed/` (extracted markdown, `chunks.json`, `chunk_analysis.json`) and `data/chroma/`.
2. Entry point: `app.py`.
3. Set `GROQ_API_KEY` in Streamlit secrets.
4. No Dockerfile and no `packages.txt` unless a system library is actually required.

First boot may download the FastEmbed ONNX weights. The index itself is already in the repo.

**Exit.** The deployed app answers from the seven documents, cites them, and both refusals still fire without a Groq call.

## Rebuild order

After any change to markdown or the manifest:

```bash
python scripts/build_chunks.py
python scripts/analyze_chunks.py
python scripts/build_index.py
```

Re-run the embedding comparison before changing `EMBEDDING_MODEL`. The current max is 893 estimated tokens, which is why Phase 5 rejects every 512-token model. If a later chunk exceeds 8,192 tokens, do not switch models silently: either keep that unit whole and accept truncation in the vector, or change the chunker and record the new cost in the README.

## Requirement trace

| Brief requirement | Closed in |
| --- | --- |
| 5–7 official documents, written prose | Phases 2–3 |
| Publisher, year, source URL, retrieval date on every document | Phases 2, 4, 6 |
| Chunks carry document name, publisher, year, section heading | Phase 4 |
| Tables and numbered recommendations stay whole | Phase 4 |
| README states the chunking choice and its cost | Phases 5, 11 |
| ChromaDB, all documents and one named document | Phases 6–7 |
| Answers only from retrieved chunks | Phases 8–9 |
| Every claim cites name, publisher, year, and a link | Phase 9 |
| Cross-document answers stay per document | Phase 9 |
| Not-in-corpus refusal in code, names what was searched | Phases 7–8 |
| Out-of-scope refusal in code | Phase 8 |
| Python, ChromaDB, embedder after analysis, Groq `openai/gpt-oss-120b`, Streamlit Cloud | Phases 1, 5, 9, 12 |

## Out of scope for this plan

- Nutrient numbers for individual foods.
- Indexing `data/raw/` error pages or page-dump `.txt` files that are not in the manifest.
- Blending two publishers inside one Groq call.
- A post-hoc checker that every sentence of model output contains a citation. The citation contract is the prompt plus the passage block. Add a checker only as a later hardening step, not as a gate for Phase 9.

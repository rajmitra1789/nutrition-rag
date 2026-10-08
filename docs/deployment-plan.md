# Deployment Plan: Streamlit Community Cloud

How to take this repository from a local folder to a public app on [Streamlit Community Cloud](https://streamlit.io/cloud). It is written against the repository as it stands today, including the gaps that will break a first deploy if left alone.

## What gets deployed

| Piece | Where it runs | Notes |
| --- | --- | --- |
| `app.py` + `src/` | Streamlit Cloud container | Entry point is `app.py` |
| Chroma index `data/chroma/` | Shipped in the repo, opened read-only at runtime | Collection `dietary_guidance`, 76 chunks |
| Corpus `src/corpus/manifest.json`, `data/processed/` | Shipped in the repo | Sidebar catalog reads the manifest |
| Embedding model `jinaai/jina-embeddings-v2-small-en` | Downloaded by FastEmbed on first query (~120 MB) | Not committed; cached in the container's temp dir |
| Generation | Groq API, `openai/gpt-oss-120b` | Needs `GROQ_API_KEY` in Streamlit secrets |

Nothing is rebuilt on the server. The index is built locally and committed. The cloud app only embeds the user's query and queries the existing collection.

Not deployed: `scripts/`, `tests/`, `data/raw/`, and `requirements-dev.txt` are harmless in the repo but are not used by the running app.

## Blockers found in the repo

| # | Blocker | Status |
| --- | --- | --- |
| 1 | **No git repository.** Streamlit Community Cloud deploys only from GitHub. | Open. Done by hand (step 1.5) |
| 2 | **Unpinned dependencies.** `requirements.txt` used `>=` ranges, but the committed index was written by `chromadb==1.5.9`. A newer Chroma may fail to open or silently migrate the SQLite file. | Fixed. Exact pins in `requirements.txt` |
| 3 | **Python version mismatch.** Local venv is 3.14.7; `runtime.txt` said 3.11.11. Community Cloud picks Python from **Advanced settings**, not `runtime.txt`. | Fixed. `runtime.txt` now records 3.12, the version to select |
| 4 | **Orphan index segment.** `data/chroma/1ca36e0b-…/` was not referenced by `chroma.sqlite3`. | Fixed. Deleted after confirming the live vector segment is `6181f694-…` |
| 5 | **Stray file.** `:memory:.ses` in the project root. | Fixed. Deleted |
| 6 | **Stale docs.** `architecture.md` and `README.md` described `bge-small-en-v1.5`, `MIN_SIMILARITY 0.36`, and 84 chunks. | Fixed. Docs match `src/config.py` and the 76-chunk index |

## Phase 1: Prepare the repository

Steps 1.1 to 1.4 are done. Steps 1.5 and 1.6 (git) are left to run by hand.

### 1.1 Clean the working tree (done)

The live folder was confirmed from `chroma.sqlite3` before anything was deleted:

```bash
sqlite3 data/chroma/chroma.sqlite3 \
  "select c.name, s.id, s.scope from segments s join collections c on c.id = s.collection;"
# dietary_guidance|6181f694-7c53-427e-b292-2741488a3614|VECTOR
# dietary_guidance|ee0ad5da-242c-47df-be1e-ad591c0968f7|METADATA
```

`1ca36e0b-…` was not referenced by any segment, so it and the stray session file were removed:

```bash
rm ":memory:.ses"
rm -rf data/chroma/1ca36e0b-3f5c-476c-b1ff-4eba6794f906
```

`data/chroma/` now holds only `chroma.sqlite3` and `6181f694-7c53-427e-b292-2741488a3614/`.

### 1.2 Pin runtime dependencies (done)

Replace `requirements.txt` with exact versions for the packages the app imports. Leave transitive packages (numpy, onnxruntime) to the resolver so pip can pick wheels for the cloud's Python version.

```text
streamlit==1.65.0
chromadb==1.5.9
fastembed==0.8.1
groq==1.7.0
python-dotenv==1.2.4
```

`pytest` moved to `requirements-dev.txt`. The app does not need it, and every extra package lengthens the cloud build.

### 1.3 Choose one Python version and rehearse with it (done, resolver check only)

Pick the version you will select in Streamlit's Advanced settings. **3.12** is the recommended choice: it is supported by all five pinned packages and avoids the 3.11 wheel gaps. Then rehearse in a clean venv with that version, because that is what the cloud will do:

```bash
python3.12 -m venv /tmp/deploy-rehearsal
source /tmp/deploy-rehearsal/bin/activate
pip install -r requirements.txt
GROQ_API_KEY=... streamlit run app.py
```

If install fails, adjust the pins here, not on the server. `runtime.txt` now says `python-3.12` to document the decision, even though Streamlit reads the version from the deploy dialog.

No Python 3.12 interpreter is installed on this machine, so instead of a full install the pins were resolved for the cloud target (Linux x86_64, Python 3.12, binary wheels only):

```bash
pip install --dry-run --ignore-installed --target /tmp/rehearse312 \
  --python-version 3.12 --platform manylinux2014_x86_64 \
  --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
  --only-binary=:all: -r requirements.txt
```

Every package resolved to a prebuilt wheel, and no `torch` or `sentence-transformers` was pulled in. A full install with a real 3.12 interpreter is still worth doing if one becomes available.

### 1.4 Verify the index with the pinned Chroma (done)

The local venv already runs `chromadb==1.5.9`, the pinned version that wrote the index, so no rebuild was needed. To check the index after any change:

```bash
python -c "from src.retrieve.index import get_collection; print(get_collection().count())"
pytest -q
```

Expected: `76`, and all tests pass. Result after cleanup: count `76`; 24 tests passed; retrieval spot checks:

| Query | Filter | Top hit | Similarity |
| --- | --- | --- | --- |
| Can I reuse cooking oil? | all | `sfa_reusing_cooking_oils` / Introduction | 0.870 |
| What should half my plate be? | `icmr_nin_my_plate` | `icmr_nin_my_plate` / My Plate for the Day food groups (all hits in that doc) | 0.800 |
| How long can I keep raw chicken in the fridge? | all | `foodsafety_cold_storage` / Fresh poultry | 0.837 |
| How do I reset my wifi router? | all | no hits; `enough=False` (not-in-corpus refusal) | — |

If you ever change the corpus, chunker, or embedding model, rebuild with `python scripts/build_chunks.py && python scripts/build_index.py` in a venv that has the pinned `chromadb`.

### 1.5 Confirm secrets are excluded

`.gitignore` already excludes `.env` and `.streamlit/secrets.toml`. Before the first commit, check that no key appears in tracked files:

```bash
git init
git add .
git status            # .env must NOT be listed
git grep -n "gsk_"    # Groq keys start with gsk_; expect no output
```

If `.env` ever ends up in a commit, rotate the key in the Groq console. Removing the file from history is not enough.

### 1.6 Push to GitHub

```bash
git commit -m "Initial commit: dietary guidance RAG app"
git branch -M main
git remote add origin git@github.com:<your-username>/nutrition-rag.git
git push -u origin main
```

The repo can be public or private. Private repos require granting Streamlit access to private repos when connecting GitHub.

Check that `data/chroma/chroma.sqlite3` and the `6181f694-…/` folder are on GitHub. The total repo is under 5 MB, well inside GitHub's limits, so Git LFS is not needed.

## Phase 2: Create the app on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the GitHub account that owns the repo.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `<your-username>/nutrition-rag`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: choose a subdomain, for example `dietary-guidance`
4. Open **Advanced settings**:
   - Python version: the version rehearsed in step 1.3 (recommended 3.12)
   - Secrets: paste the block below
5. Click **Deploy**.

```toml
GROQ_API_KEY = "gsk_your_real_key"
```

`src/config.py` reads `GROQ_API_KEY` from the environment first, then from `st.secrets`, so this is the only secret the app needs. All other settings (`GROQ_MODEL`, `MIN_SIMILARITY`, `TOP_K`, `EMBEDDING_MODEL`, `GROQ_MAX_TOKENS`) have working defaults in `src/config.py`. Do not override them in the cloud unless you have changed them locally and rebuilt the index to match. Changing `EMBEDDING_MODEL` without rebuilding the index will break retrieval.

The build installs `requirements.txt` and starts the app. Expect a few minutes for the first build.

## Phase 3: Verify the live app

Watch **Manage app → Logs** during the first boot, then run these checks in the live app.

| Check | How | Expected |
| --- | --- | --- |
| App boots | Open the URL | Title "Dietary Guidance Assistant", sidebar lists 7 documents |
| Model download | First question | Slow (model download), then an answer. Later questions are fast |
| In-corpus answer | "Can I reuse cooking oil?" | Answer from the SFA document with a `(Document, Publisher, Year)` citation |
| Per-document filter | Select "ICMR-NIN My Plate" and ask "What should half my plate be?" | Only ICMR passages in "Retrieved passages" |
| Out-of-scope refusal | "How many calories should I eat to lose weight?" | Refusal pointing to a qualified professional; no passages shown |
| Not-in-corpus refusal | "How do I reset my wifi router?" | Refusal that names the documents searched |
| Missing key handling | (Optional) temporarily remove the secret | Error message in chat, not a crash |

If the cooking-oil question returns the not-in-corpus refusal, the index and the embedding model disagree. See Troubleshooting.

## Phase 4: Operate the app

### Updating code

Every push to `main` redeploys automatically. Keep the habit: rehearse locally in the pinned venv, run `pytest -q`, then push.

### Updating the corpus or chunking

The cloud never rebuilds the index, so any change to `data/processed/extracted/`, `src/corpus/manifest.json`, `src/ingest/chunk.py`, or `EMBEDDING_MODEL` needs a local rebuild before the push:

```bash
python scripts/build_chunks.py
python scripts/analyze_chunks.py
python scripts/build_index.py
git add data/processed data/chroma src/corpus
git commit -m "Rebuild index: <what changed>"
git push
```

After a push that changes `data/chroma/`, use **Manage app → Reboot app** so the cached collection (`get_collection` is `lru_cache`d per process) is reopened from the new files.

### Groq free-plan limits

From `src/config.py`: 30 requests/minute, 1,000 requests/day, 8,000 tokens/minute, 200,000 tokens/day. A cross-document question makes **one Groq call per matching document**, up to `MAX_DOCS_IN_ANSWER = 5`. A broad question measures about 2,800 tokens, so a busy public app can hit the 8,000 tokens/minute limit at roughly 3 broad questions per minute. When the limit is hit, the user sees "I could not complete that request: …" in the chat. If the app gets real traffic, move to a paid Groq tier or lower `MAX_DOCS_IN_ANSWER`.

### Sleep and cold starts

Community Cloud puts apps to sleep after a period without traffic. Waking it restarts the container, which means the embedding model is downloaded again on the next first question. That is expected; the first answer after a wake-up is slow.

### Resource limits

Community Cloud containers have about 1 GB of RAM. The app's footprint is small: a 33M-parameter ONNX embedder, a 76-row Chroma collection, no PyTorch. It should fit comfortably. If logs show the app being killed for memory, check that no dependency has pulled in `torch` or `sentence-transformers`.

## Rollback

- **Bad code push:** `git revert <commit>` and push. The app redeploys from the reverted state.
- **Bad index push:** revert the commit that changed `data/chroma/`, push, then reboot the app.
- **Leaked key:** rotate it in the Groq console, update **Settings → Secrets** in Streamlit, and reboot.

## Troubleshooting

| Symptom in logs or app | Likely cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: src` | App started from a different working directory | Already handled: `app.py` adds the repo root to `sys.path`. Check the main file path is `app.py` at repo root |
| pip cannot find a wheel during build | Python version in Advanced settings differs from the rehearsal | Delete the app and redeploy with the rehearsed version (the Python version cannot be changed on an existing app) |
| `sqlite3` version error from Chroma | Container SQLite older than Chroma requires | Add `pysqlite3-binary` to `requirements.txt` and swap it in at the top of `app.py` before importing Chroma |
| Chroma error opening `chroma.sqlite3` | Index written by a different Chroma version | Rebuild locally with the pinned `chromadb`, commit, push, reboot |
| Every question returns not-in-corpus | Collection is empty, or embedding model differs from the one used to build | Confirm `data/chroma/` was pushed; confirm `EMBEDDING_MODEL` is not overridden in secrets |
| "I could not complete that request: … API key" | Secret missing or misnamed | Secret must be exactly `GROQ_API_KEY` at the top level of the TOML |
| "… rate limit" errors | Groq free-plan cap | Wait, upgrade the Groq plan, or lower `MAX_DOCS_IN_ANSWER` |
| First answer takes 30+ seconds | FastEmbed downloading the model after a cold start | Expected once per container start |

## Checklist

- [x] Remove `:memory:.ses` and the orphan Chroma segment
- [x] Pin `requirements.txt`; move `pytest` to dev requirements
- [x] Choose a Python version, update `runtime.txt`, resolve pins for Linux/3.12
- [x] Verify the index with pinned Chroma; count is 76; `pytest -q` passes
- [ ] `git init`, confirm `.env` is not staged, push to GitHub
- [ ] Create the Streamlit app: `app.py`, chosen Python version, `GROQ_API_KEY` secret
- [ ] Run the Phase 3 checks on the live URL
- [x] Update `README.md` deploy section and `docs/architecture.md` to match

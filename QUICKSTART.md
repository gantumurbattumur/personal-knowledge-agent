# Personal Knowledge Agent — Quickstart

## 1) Start using it

```bash
# health check
rag doctor

# see connected sources
rag sources list

# index a folder immediately
rag build ~/Documents

# query
rag search "neural networks"
rag ask "summarize key ideas from my ML notes"
```

---

## 2) If you want to add more folders to be indexed

You can connect multiple local folders. Each folder is treated as a source.

```bash
# connect one folder
rag sources connect local --path ~/Documents --source-id docs

# connect another folder
rag sources connect local --path ~/Notes --source-id notes

# verify
rag sources list
```

Then build index for each folder (or whichever changed):

```bash
rag build ~/Documents
rag build ~/Notes
```

Notes:
- `sources connect` registers the source metadata.
- `rag build <path>` performs actual parsing/chunking/indexing for that path.

---

## 3) What happens if files change in indexed folders?

When you run `rag build <path>` again:
- **Unchanged chunks** are skipped.
- **Changed files/chunks** are updated.
- **New files** are added.

Current behavior to know:
- If a file is renamed/deleted, old indexed chunks may remain until you do a clean rebuild strategy.

Recommended clean-rebuild pattern:

```bash
# build a clean shadow generation
rag build-generation v2-refresh --embedding-profile-id v1-minilm --chunking-profile-id v1-fixed --paths ~/Documents,~/Notes

# inspect
rag list-generations

# switch to new generation when satisfied
rag activate-generation v2-refresh
```

---

## 4) What to do in the future to scale up

### A. Operational scale (more data, more sources)
- Keep each corpus as its own connected source (`--source-id` helps management).
- Use `rag sync` for cloud sources (Google Drive / Notion), then query normally.
- Rebuild incrementally on a schedule (daily/hourly), and do periodic clean generation rebuilds.

### B. Retrieval quality scale
- Keep reranker enabled (`rag doctor` should show reranker `enabled`).
- Prefer external corpus sources over repo files (already enabled in your current setup).
- Move to stronger embeddings with generation workflow:

```bash
# set OPENAI_API_KEY in .env first
rag build-generation v2-openai --embedding-profile-id openai-large --chunking-profile-id v1-fixed --paths ~/Documents,~/Notes
rag activate-generation v2-openai
```

### C. Safety / release hygiene
- Never commit credential files (`client_secret.json`, `.env` secrets).
- Keep secrets only in `.env` or secret manager.
- Rotate keys immediately if they are ever exposed.

---

## 5) Minimal daily workflow

```bash
rag sync
rag build ~/Documents
rag ask "what changed in my AI notes this week?"
```

If results look stale/noisy, rebuild a new generation and activate it.

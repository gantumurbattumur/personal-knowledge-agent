# personal-knowledge-agent

Workflow-aware, local-first terminal RAG assistant for developers.

## Features (v1 scaffold)

- Local ingestion for Markdown, code/text files, PDF, and EPUB
- Project-local storage under `.pka/` (SQLite metadata + LanceDB vectors)
- Incremental chunk indexing into SQLite + FTS5
- Optional dense embeddings + hybrid retrieval fusion (RRF)
- Optional reranking for top candidates
- Project workflow context capture (cwd, git root, git branch, recent files)
- Workflow-aware score boosting (+50% for CWD / matching branch sources)
- Cloud connectors: Google Drive incremental sync (version checks), Notion markdown sync
- Docling-first PDF extraction; optional image/chart captioning is file-based (no camera capture)
- Grounded retrieval with source references
- Terminal CLI for `rag init`, `rag auth`, `rag sync`, `rag ask`, `rag doctor`, `rag trace`

## Quick start

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Install workflow + connector stack:

```bash
uv pip install -e .[full]
```

Install full ingestion + hybrid retrieval extras:

```bash
uv pip install -e .[full]
```

Install only Google Drive support:

```bash
uv pip install -e .[gdrive]
```

Initialize project-local config and env:

```bash
rag init
```

Index your project/documents:

```bash
rag index .
```

Search the index:

```bash
rag search "colbert reranking"
```

Search with explicit mode override:

```bash
rag search "colbert reranking" --mode hybrid
```

Ask a grounded question:

```bash
rag ask "What did I write about retrieval evaluation?"
```

Ask using lexical-only fallback:

```bash
rag ask "What did I write about retrieval evaluation?" --mode lexical
```

## Cloud auth and sync

Authenticate connectors (tokens stored in system keyring):

```bash
rag auth google-drive --client-secret-file client_secret.json
```

For Notion, first print authorization URL, then exchange code:

```bash
rag auth notion --client-id <id>
rag auth notion --client-id <id> --client-secret <secret> --code <oauth_code>
```

Run incremental sync:

```bash
rag sync
```

`rag sync` reads `.env` keys:

- `GOOGLE_DRIVE_FOLDER_ID`
- `GOOGLE_DRIVE_DEST` (default `.pka/sources/google-drive`)
- `NOTION_ROOT_PAGE_ID`
- `NOTION_DEST` (default `.pka/sources/notion`)

Only changed cloud documents are pulled (Google Drive by `version`, Notion by `last_edited_time`). Synced files are indexed automatically.

## Google Drive workflow

Connect your Drive folder once, then use regular sync/index flow:

```bash
rag sources connect google_drive --folder-id <FOLDER_ID>
rag sync
```

By default files are stored under `.pka/sources/google-drive` and are indexed incrementally on each `rag sync`.

## LLM modes

- `none` (default): extractive grounded answer from retrieved chunks
- `ollama`: calls local Ollama (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`)
- `openai`: calls OpenAI Chat Completions (`OPENAI_API_KEY`, `OPENAI_MODEL`)

Set provider in `.pka/config.toml`.

## Retrieval modes

- `lexical`: SQLite FTS5 only (fastest, zero ML deps)
- `dense`: embeddings vector search only
- `hybrid` (default): lexical + dense fusion

Dense/hybrid requires optional dependency set:

```bash
uv pip install -e .[retrieval]
```

## Vector database status

Current implementation uses SQLite as local storage:

- FTS5 table for lexical retrieval
- `chunk_embeddings` table storing dense vectors as blobs
- Dense retrieval uses in-process cosine similarity scan

This is local vector storage in SQLite, not an external dedicated vector database yet.

Default retrieval config in `.pka/config.toml`:

```toml
[retrieval]
mode = "hybrid"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
enable_reranker = false
reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

## Index Generations & Shadow Builds

Build shadow index generations to test new embedding models/chunking strategies without affecting active retrieval:

```bash
# Build a shadow index with OpenAI embeddings
rag build-generation v2-openai --embedding-profile-id openai-large

# List all generations
rag list-generations

# Activate when satisfied
rag activate-generation v2-openai
```

## Query Normalization & Enhancements

- **Query normalization**: Automatic typo fixing and grammatical error correction before retrieval
- **Reranker**: Cross-encoder rescoring for precision improvement (always enabled by default)
- **Multi-provider embeddings**: Local (sentence-transformers), OpenAI, or Jina embeddings
- **Chunking backends**: Pluggable chunk creation (fixed-size, semantic, Docling-aware)

## Architecture modules

- `personal_knowledge_agent/schemas.py`: core data contracts
- `personal_knowledge_agent/ingest.py`: file discovery + parsing + chunking (pluggable backends)
- `personal_knowledge_agent/index_store.py`: SQLite storage + FTS5 + generation management
- `personal_knowledge_agent/embeddings.py`: multi-provider embedding + reranker engines
- `personal_knowledge_agent/retrieval.py`: lexical/dense/hybrid + context fusion + reranking
- `personal_knowledge_agent/llm.py`: LLM provider abstraction + query normalization
- `personal_knowledge_agent/cli.py`: terminal UX + generation commands
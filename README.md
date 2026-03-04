# personal-knowledge-agent

Local-first, terminal-based personal RAG assistant for developers.

## What it does

- Indexes local project/docs into a local `.pka/` store.
- Retrieves relevant chunks with lexical/dense/hybrid modes.
- Answers questions with grounded references in the terminal.
- Optionally syncs from Google Drive / Notion connectors.

## Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

For full local + cloud stack:

```bash
uv pip install -e .[full]
```

## Local quick start

1) Initialize project settings:

```bash
rag init
```

2) Build index from current folder (or any path):

```bash
rag build .
```

3) Ask a question:

```bash
rag ask "How does retrieval work in this project?"
```

## Core commands

```bash
rag init
rag build <path>
rag index <path>     # alias of build
rag search "<query>"
rag ask "<question>"
rag doctor
rag status
rag trace "<query>"
```

## Connector commands (optional)

```bash
rag sources list
rag sources connect local --path /path/to/docs
rag sources connect google_drive --folder-id <FOLDER_ID>
rag auth google-drive --client-secret-file client_secret.json
rag sync
rag drive auth|sync|status
```

## Notes

- Use `rag --help` to view all command groups and options.
- `rag build` / `rag index` is the current indexing path (not `rag ingest`).
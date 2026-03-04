# Personal Knowledge Agent — Quickstart

## 1) Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Optional full extras (connectors + retrieval stack):

```bash
uv pip install -e .[full]
```

---

## 2) Initialize

```bash
rag init
rag doctor
```

---

## 3) Build index from your workspace

```bash
rag build .
```

This indexes your files into the local project index database.

---

## 4) Ask questions

```bash
rag ask "How does retrieval work in this codebase?"
```

---

## 5) Search-only mode (optional)

```bash
rag search "retrieval pipeline"
```

---

## 6) Google Drive (optional)

Authenticate:

```bash
rag drive auth --client-secret-file client_secret.json
```

Sync and index Drive files:

```bash
rag drive sync
rag drive status
```

---

## 7) Useful command map

```bash
rag init
rag build <path>
rag index <path>           # alias
rag search "<query>"
rag ask "<question>"
rag doctor
rag status
rag sources list
rag sources connect <type> ...
rag drive auth|sync|status
```

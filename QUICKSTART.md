# Personal Knowledge Agent — Quickstart (Version 2)

## 1) Install and initialize

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .[daemon]
rag init
rag doctor
```

For full connector support:

```bash
uv pip install -e .[full]
```

---

## 2) Register and index your workspace

```bash
rag init-workspace .
rag workspace list
```

This creates/updates registry state in `~/.pka/workspaces.json` and indexes into `.pka/index.sqlite3`.

---

## 3) Start the V2 daemon

```bash
rag daemon start
```

In another terminal:

```bash
rag daemon status
curl -s http://127.0.0.1:8741/health
```

---

## 4) Read daemon token for clients

```bash
cat ~/.pka/daemon_token
```

Chrome and VS Code clients must send this token as `X-RAG-Token`.

---

## 5) Query with workspace context

### CLI path

```bash
rag ask "How does auth middleware work?"
```

### Daemon API path

```bash
TOKEN=$(cat ~/.pka/daemon_token)
curl -s -X POST http://127.0.0.1:8741/query \
	-H "Content-Type: application/json" \
	-H "X-RAG-Token: $TOKEN" \
	-d '{
		"query": "How does auth middleware work?",
		"workspace_hint": "'"$(pwd)"'"
	}'
```

---

## 6) Google Drive in V2

Authenticate:

```bash
rag drive auth --client-secret-file client_secret.json
```

Sync and index Drive workspace:

```bash
rag drive sync
rag drive status
```

Drive is handled as `drive://default` and stored separately from local repo workspaces.

---

## 7) Client setup pointers

- Chrome extension files: `clients/chrome-extension/`
- VS Code extension files: `clients/vscode-extension/`

Both use daemon `http://localhost:8741` and V2 request context fields:

- `workspace_hint`
- `active_file`
- `active_code_selection`
- `clipboard_context`

---

## 8) Useful V2 commands

```bash
rag daemon start|stop|status
rag init-workspace <path>
rag workspace list|remove
rag drive auth|sync|status
```

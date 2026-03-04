# personal-knowledge-agent

Workspace-aware, local-first RAG assistant for developers.

## Version 2 (V2)

V2 upgrades the project from a terminal-only tool to a persistent local assistant with a daemon API and multi-client access.

### What’s new in V2

- FastAPI daemon on `127.0.0.1:8741`
- Workspace registry persisted at `~/.pka/workspaces.json`
- Workspace-aware routing from `workspace_hint` or `active_file`
- Shared request schema across CLI, Chrome, and VS Code clients
- Local auth token (`~/.pka/daemon_token`) required via `X-RAG-Token`
- Chrome extension client (`clients/chrome-extension/`)
- VS Code extension client (`clients/vscode-extension/`)
- Dedicated Drive workspace (`drive://default`) and connector abstraction

## V2 architecture

```text
Chrome Extension      VSCode Extension      CLI
			│                     │                │
			└──────────────┬──────┴────────────────┘
										 ▼
					FastAPI Daemon (localhost:8741)
										 │
			 WorkspaceRegistry + RAG Core Pipeline
										 │
			per-workspace SQLite + vector stores
```

## Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .[daemon]
```

For full local + cloud stack:

```bash
uv pip install -e .[full]
```

## V2 quick usage

1. Initialize project settings

```bash
rag init
```

2. Register and index a workspace

```bash
rag init-workspace .
```

3. Start daemon

```bash
rag daemon start
```

4. Check daemon and workspace state

```bash
rag daemon status
rag workspace list
```

5. Query via CLI (core path still supported)

```bash
rag ask "How does retrieval routing work in V2?"
```

## New CLI commands in V2

```bash
rag daemon start|stop|status
rag init-workspace <path>
rag workspace list|remove
rag drive auth|sync|status
```

## Daemon API (V2)

- `GET /health`
- `POST /query`
- `POST /ingest`
- `GET /workspaces`
- `POST /workspaces`

`POST /query` accepts:

```json
{
	"query": "string",
	"workspace_hint": "/absolute/path/or/drive://default",
	"clipboard_context": "optional user-selected text",
	"active_file": "/absolute/path/to/file",
	"active_code_selection": "optional highlighted code"
}
```

## Security (local daemon)

- Token file: `~/.pka/daemon_token`
- Clients must send: `X-RAG-Token: <token>`

## Client locations

- Chrome extension: `clients/chrome-extension/`
- VS Code extension: `clients/vscode-extension/`

## Notes

- Existing RAG core modules remain intact (`retrieve_with_context`, `generate_answer`, `ingest_path`).
- V2 wraps and exposes the core through a persistent daemon and client integrations.
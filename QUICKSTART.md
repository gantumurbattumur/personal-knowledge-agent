# Personal Knowledge Agent — Quickstart

## 1) One-time setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .[full]
rag init
rag doctor
```

---

## 2) Connect multiple local folders

Use unique `--source-id` values for each local folder.

```bash
rag sources connect local --path ~/Documents --source-id docs
rag sources connect local --path ~/Notes --source-id notes
rag sources connect local --path ~/Research --source-id research
rag sources list
```

First indexing run:

```bash
rag build ~/Documents
rag build ~/Notes
rag build ~/Research
```

---

## 3) Connect multiple cloud sources

### Google Drive (multiple folders)

Authenticate once:

```bash
rag auth google-drive --client-secret-file client_secret.json
```

Connect one or more Drive folders:

```bash
rag sources connect google_drive --folder-id <FOLDER_ID_1> --destination .pka/sources/google-drive/work --source-id gdrive-work
rag sources connect google_drive --folder-id <FOLDER_ID_2> --destination .pka/sources/google-drive/personal --source-id gdrive-personal
```

### Notion (multiple roots)

```bash
rag auth notion --client-id <id>
rag auth notion --client-id <id> --client-secret <secret> --code <oauth_code>
rag sources connect notion --root-page-id <ROOT_PAGE_ID_1> --source-id notion-main
rag sources connect notion --root-page-id <ROOT_PAGE_ID_2> --source-id notion-wiki
```

Sync all connected cloud sources:

```bash
rag sync
rag sources list
```

---

## 4) Incremental updates (new files only)

Day-to-day update commands:

```bash
# Cloud: pulls + indexes only new/changed cloud docs
rag sync

# Local: indexes only new/changed local files for each path
rag build ~/Documents
rag build ~/Notes
rag build ~/Research
```

Behavior summary:
- `rag sync` is incremental for all connected cloud sources.
- `rag build <path>` is incremental for that local path.
- Unchanged chunks are skipped automatically.

---

## 5) Query and verify

```bash
rag search "neural networks"
rag ask "summarize key ideas from my ML notes"
rag status
```

---

## 6) If you deleted/renamed many files and want strict cleanup

Incremental mode handles new/changed content well. For major deletions/renames, do a full reset:

```bash
rm -rf .pka
rag init
rag sources list
rag sync
rag build ~/Documents
rag build ~/Notes
```

Reconnect any sources if needed, then re-run `rag sync` and `rag build`.

"""Workspace registry — maps workspace paths to their vector stores and settings.

Persisted at ``~/.pka/workspaces.json``.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from personal_knowledge_agent.config import Settings, load_settings
from personal_knowledge_agent.index_store import IndexStore
from personal_knowledge_agent.vector_store import LanceVectorStore

REGISTRY_PATH = Path.home() / ".pka" / "workspaces.json"


@dataclass
class WorkspaceHandle:
    """Runtime handle for a loaded workspace."""

    id: str
    path: str
    kind: str  # "local" | "drive"
    db_path: str
    last_indexed: Optional[str] = None
    # lazy-loaded runtime objects ------------------------------------------------
    _settings: Optional[Settings] = field(default=None, repr=False)
    _store: Optional[IndexStore] = field(default=None, repr=False)
    _vector_store: Optional[LanceVectorStore] = field(default=None, repr=False)

    # -- lazy accessors ----------------------------------------------------------

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            if self.kind == "local":
                self._settings = load_settings(Path(self.path))
            else:
                # Drive workspaces use the global home settings
                self._settings = load_settings(Path.home())
        return self._settings

    @property
    def store(self) -> IndexStore:
        if self._store is None:
            self._store = IndexStore(Path(self.db_path))
        return self._store

    @property
    def vector_store(self) -> LanceVectorStore:
        if self._vector_store is None:
            vs_path = Path(self.db_path).parent / "lancedb"
            self._vector_store = LanceVectorStore(vs_path)
        return self._vector_store

    def close(self) -> None:
        if self._store is not None:
            try:
                self._store.__exit__(None, None, None)
            except Exception:
                pass
            self._store = None
        self._vector_store = None
        self._settings = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "kind": self.kind,
            "db_path": self.db_path,
            "last_indexed": self.last_indexed,
        }


class WorkspaceRegistry:
    """Persistent registry of known workspaces."""

    def __init__(self, workspaces: list[WorkspaceHandle] | None = None) -> None:
        self._workspaces: dict[str, WorkspaceHandle] = {}
        for ws in (workspaces or []):
            self._workspaces[ws.path] = ws

    # -- persistence -------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> WorkspaceRegistry:
        registry_path = path or REGISTRY_PATH
        if not registry_path.exists():
            return cls()
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        handles = []
        for entry in data.get("workspaces", []):
            handles.append(
                WorkspaceHandle(
                    id=entry.get("id", f"ws_{uuid.uuid4().hex[:8]}"),
                    path=entry["path"],
                    kind=entry.get("kind", "local"),
                    db_path=entry["db_path"],
                    last_indexed=entry.get("last_indexed"),
                )
            )
        return cls(handles)

    def save(self, path: Path | None = None) -> None:
        registry_path = path or REGISTRY_PATH
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"workspaces": [ws.to_dict() for ws in self._workspaces.values()]}
        registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -- mutations ---------------------------------------------------------------

    def register(self, ws_path: str, kind: str = "local") -> WorkspaceHandle:
        """Register a workspace. Returns the handle (existing or new)."""
        if ws_path in self._workspaces:
            return self._workspaces[ws_path]

        if kind == "drive":
            db_dir = Path.home() / ".pka" / "drive"
            db_path = str(db_dir / "index.sqlite3")
        else:
            db_path = str(Path(ws_path) / ".pka" / "index.sqlite3")

        handle = WorkspaceHandle(
            id=f"ws_{uuid.uuid4().hex[:8]}",
            path=ws_path,
            kind=kind,
            db_path=db_path,
            last_indexed=datetime.now(timezone.utc).isoformat(),
        )
        self._workspaces[ws_path] = handle
        self.save()
        return handle

    def unregister(self, ws_path: str) -> bool:
        handle = self._workspaces.pop(ws_path, None)
        if handle:
            handle.close()
            self.save()
            return True
        return False

    # -- lookups -----------------------------------------------------------------

    def get(self, ws_path: str) -> WorkspaceHandle | None:
        return self._workspaces.get(ws_path)

    def find_nearest(self, file_path: Path) -> WorkspaceHandle | None:
        """Walk up the directory tree to find the nearest registered workspace root."""
        resolved = file_path.resolve()
        # Check the file itself (in case it's a directory that's registered)
        for candidate in [resolved] + list(resolved.parents):
            ws = self._workspaces.get(str(candidate))
            if ws is not None:
                return ws
        return None

    def list_all(self) -> list[dict]:
        return [ws.to_dict() for ws in self._workspaces.values()]

    def loaded_count(self) -> int:
        return len(self._workspaces)

    # -- cleanup -----------------------------------------------------------------

    def close_all(self) -> None:
        for ws in self._workspaces.values():
            ws.close()

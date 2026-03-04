"""Authentication middleware — generates & validates a per-machine daemon token.

The token is stored at ``~/.pka/daemon_token`` and sent by every client as the
``X-RAG-Token`` header.
"""
from __future__ import annotations

import secrets
from pathlib import Path

TOKEN_PATH = Path.home() / ".pka" / "daemon_token"


def ensure_token() -> str:
    """Return the daemon token, creating one if it doesn't exist."""
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TOKEN_PATH.exists():
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    TOKEN_PATH.write_text(token, encoding="utf-8")
    TOKEN_PATH.chmod(0o600)
    return token


def read_token() -> str | None:
    """Read the daemon token from disk (returns None if missing)."""
    if TOKEN_PATH.exists():
        return TOKEN_PATH.read_text(encoding="utf-8").strip() or None
    return None

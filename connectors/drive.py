"""Google Drive connector — higher-level wrapper around personal_knowledge_agent.drive.

Handles authentication, incremental sync, and automatic re-indexing into the
Drive workspace's dedicated SQLite database.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from personal_knowledge_agent.auth import auth_google_drive, load_token
from personal_knowledge_agent.drive import (
    IncrementalSyncResult,
    sync_google_drive_incremental,
)
from personal_knowledge_agent.index_store import IndexStore
from personal_knowledge_agent.ingest import ingest_path

logger = logging.getLogger("pka.connectors.drive")

# Supported file extensions for indexing after download
INDEXABLE_EXTENSIONS = {".txt", ".md", ".pdf", ".csv", ".json", ".yaml", ".yml", ".toml", ".py", ".js", ".ts"}

DEFAULT_DRIVE_DB = Path.home() / ".pka" / "drive" / "index.sqlite3"
DEFAULT_DRIVE_DEST = Path.home() / ".pka" / "sources" / "google-drive"


@dataclass
class DriveConnector:
    """High-level connector for Google Drive sync + indexing."""

    folder_id: str = ""
    destination: Path = field(default_factory=lambda: DEFAULT_DRIVE_DEST)
    db_path: Path = field(default_factory=lambda: DEFAULT_DRIVE_DB)
    client_secret_file: str = "client_secret.json"

    # -- Authentication ---

    def authenticate(self) -> bool:
        """Run the OAuth2 flow. Returns True on success."""
        result = auth_google_drive(client_secret_file=self.client_secret_file)
        if result.ok:
            logger.info("Drive authentication successful: %s", result.message)
        else:
            logger.error("Drive authentication failed: %s", result.message)
        return result.ok

    def is_authenticated(self) -> bool:
        token = load_token("google-drive")
        return token is not None

    # -- Sync ---

    def sync(self) -> IncrementalSyncResult:
        """Download changed files from Drive and re-index them."""
        token = load_token("google-drive")
        if not token:
            return IncrementalSyncResult(
                ok=False,
                destination=self.destination,
                downloaded=0,
                unchanged=0,
                message="Not authenticated. Run DriveConnector.authenticate() first.",
            )

        folder_id = self.folder_id or os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")
        if not folder_id:
            return IncrementalSyncResult(
                ok=False,
                destination=self.destination,
                downloaded=0,
                unchanged=0,
                message="No folder_id configured. Set GOOGLE_DRIVE_FOLDER_ID.",
            )

        self.destination.mkdir(parents=True, exist_ok=True)

        logger.info("Starting Drive sync: folder=%s → %s", folder_id, self.destination)

        with IndexStore(self.db_path) as store:
            result = sync_google_drive_incremental(
                folder_id=folder_id,
                destination=self.destination,
                token_payload=token,
                store=store,
            )

        if result.ok and result.downloaded > 0:
            self._reindex()

        return result

    # -- Status ---

    def status(self) -> dict:
        """Return sync status information."""
        info: dict = {
            "authenticated": self.is_authenticated(),
            "folder_id": self.folder_id or os.environ.get("GOOGLE_DRIVE_FOLDER_ID", ""),
            "destination": str(self.destination),
            "db_path": str(self.db_path),
            "db_exists": self.db_path.exists(),
        }

        if self.db_path.exists():
            with IndexStore(self.db_path) as store:
                stats = store.stats()
                sync_state = store.get_sync_state("google_drive")
            info["chunks"] = stats["chunk_count"]
            info["sources"] = stats["source_count"]
            info["last_sync"] = sync_state.get("last_synced")
            info["db_size_kb"] = round(self.db_path.stat().st_size / 1024, 1)
        else:
            info["chunks"] = 0
            info["sources"] = 0

        return info

    # -- Internal ---

    def _reindex(self) -> None:
        """Re-index all downloaded files into the Drive workspace DB."""
        logger.info("Re-indexing Drive files at %s", self.destination)
        bundle = ingest_path(self.destination)
        if not bundle.chunks:
            logger.info("No chunks produced from Drive files")
            return

        with IndexStore(self.db_path) as store:
            inserted, skipped, _updated = store.upsert_chunks(bundle.chunks)
            assets = (
                store.upsert_assets([a.model_dump() for a in bundle.assets])
                if bundle.assets
                else 0
            )
        logger.info(
            "Drive re-index complete: inserted=%d, skipped=%d, assets=%d",
            inserted,
            skipped,
            assets,
        )

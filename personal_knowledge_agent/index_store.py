from __future__ import annotations

import math
import sqlite3
from array import array
from pathlib import Path

from .schemas import Chunk, RetrievalResult


class IndexStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_type TEXT NOT NULL,
                title TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                section TEXT NOT NULL,
                text TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                connector TEXT NOT NULL DEFAULT 'local',
                page_number INTEGER,
                bounding_box_json TEXT,
                cloud_url TEXT,
                image_description TEXT,
                external_id TEXT,
                version_id TEXT,
                branch_hint TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                title,
                section,
                text,
                tokenize='porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS query_trace (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                query TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                provider TEXT NOT NULL,
                retrieval_mode TEXT,
                git_branch TEXT,
                cwd TEXT,
                result_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                dims INTEGER NOT NULL,
                vector_blob BLOB NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY(chunk_id, embedding_model),
                FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_fingerprint ON chunks(fingerprint);
            CREATE INDEX IF NOT EXISTS idx_embeddings_model ON chunk_embeddings(embedding_model);

            CREATE TABLE IF NOT EXISTS sync_state (
                connector TEXT NOT NULL,
                external_id TEXT NOT NULL,
                version_id TEXT,
                source_path TEXT,
                cloud_url TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY(connector, external_id)
            );

            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                title TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                page_number INTEGER,
                file_path TEXT,
                bbox_json TEXT,
                caption_text TEXT,
                ocr_text TEXT,
                figure_id TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS page_maps (
                source_path TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                layout_json TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY(source_path, page_number)
            );

            CREATE TABLE IF NOT EXISTS source_connections (
                source_type TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1,
                account_label TEXT,
                config_json TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        self._migrate_schema()
        self.conn.commit()

    def _migrate_schema(self) -> None:
        chunk_columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "connector" not in chunk_columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN connector TEXT NOT NULL DEFAULT 'local'")
        if "page_number" not in chunk_columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN page_number INTEGER")
        if "bounding_box_json" not in chunk_columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN bounding_box_json TEXT")
        if "cloud_url" not in chunk_columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN cloud_url TEXT")
        if "image_description" not in chunk_columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN image_description TEXT")
        if "external_id" not in chunk_columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN external_id TEXT")
        if "version_id" not in chunk_columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN version_id TEXT")
        if "branch_hint" not in chunk_columns:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN branch_hint TEXT")

        trace_columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(query_trace)").fetchall()
        }
        if "retrieval_mode" not in trace_columns:
            self.conn.execute("ALTER TABLE query_trace ADD COLUMN retrieval_mode TEXT")

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_state (
                connector TEXT NOT NULL,
                external_id TEXT NOT NULL,
                version_id TEXT,
                source_path TEXT,
                cloud_url TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY(connector, external_id)
            )
            """
        )

        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_external_id ON chunks(external_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_version_id ON chunks(version_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_source_path ON assets(source_path)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(asset_type)")

    def __enter__(self) -> "IndexStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def upsert_chunks(self, chunks: list[Chunk]) -> tuple[int, int, list[Chunk]]:
        inserted = 0
        skipped = 0
        updated_chunks: list[Chunk] = []

        for chunk in chunks:
            existing = self.conn.execute(
                "SELECT fingerprint FROM chunks WHERE chunk_id = ?",
                (chunk.chunk_id,),
            ).fetchone()

            if existing and existing["fingerprint"] == chunk.fingerprint:
                skipped += 1
                continue

            self.conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, source_path, source_type, title, fingerprint, section, text,
                    token_estimate, updated_at, connector, page_number, bounding_box_json,
                    cloud_url, image_description, external_id, version_id, branch_hint
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    source_path=excluded.source_path,
                    source_type=excluded.source_type,
                    title=excluded.title,
                    fingerprint=excluded.fingerprint,
                    section=excluded.section,
                    text=excluded.text,
                    token_estimate=excluded.token_estimate,
                    updated_at=excluded.updated_at,
                    connector=excluded.connector,
                    page_number=excluded.page_number,
                    bounding_box_json=excluded.bounding_box_json,
                    cloud_url=excluded.cloud_url,
                    image_description=excluded.image_description,
                    external_id=excluded.external_id,
                    version_id=excluded.version_id,
                    branch_hint=excluded.branch_hint
                """,
                (
                    chunk.chunk_id,
                    chunk.source_path,
                    chunk.source_type,
                    chunk.title,
                    chunk.fingerprint,
                    chunk.section,
                    chunk.text,
                    chunk.token_estimate,
                    chunk.updated_at,
                    chunk.connector,
                    chunk.metadata.page_number,
                    chunk.metadata.bounding_box.model_dump_json() if chunk.metadata.bounding_box else None,
                    chunk.metadata.cloud_url,
                    chunk.metadata.image_description,
                    chunk.metadata.external_id,
                    chunk.metadata.version_id,
                    chunk.metadata.branch_hint,
                ),
            )
            self.conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                (chunk.chunk_id,),
            )
            self.conn.execute(
                "INSERT INTO chunks_fts (chunk_id, title, section, text) VALUES (?, ?, ?, ?)",
                (chunk.chunk_id, chunk.title, chunk.section, chunk.text),
            )
            inserted += 1
            updated_chunks.append(chunk)

        self.conn.commit()
        return inserted, skipped, updated_chunks

    def chunks_missing_embeddings(self, embedding_model: str) -> list[Chunk]:
        rows = self.conn.execute(
            """
            SELECT c.chunk_id, c.source_path, c.source_type, c.title, c.fingerprint, c.section, c.text, c.token_estimate, c.updated_at, c.connector
            FROM chunks c
            LEFT JOIN chunk_embeddings e
                ON e.chunk_id = c.chunk_id AND e.embedding_model = ?
            WHERE e.chunk_id IS NULL
            """,
            (embedding_model,),
        ).fetchall()
        return [
            Chunk(
                chunk_id=row["chunk_id"],
                source_path=row["source_path"],
                source_type=row["source_type"],
                title=row["title"],
                fingerprint=row["fingerprint"],
                section=row["section"],
                text=row["text"],
                token_estimate=int(row["token_estimate"]),
                updated_at=row["updated_at"],
                connector=row["connector"] if "connector" in row.keys() else "local",
            )
            for row in rows
        ]

    def upsert_embeddings(self, embedding_model: str, chunk_ids: list[str], vectors: list[list[float]]) -> int:
        if not chunk_ids or not vectors:
            return 0

        count = 0
        for chunk_id, vector in zip(chunk_ids, vectors):
            buffer = array("f", vector).tobytes()
            dims = len(vector)
            self.conn.execute(
                """
                INSERT INTO chunk_embeddings (chunk_id, embedding_model, dims, vector_blob, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(chunk_id, embedding_model) DO UPDATE SET
                    dims=excluded.dims,
                    vector_blob=excluded.vector_blob,
                    updated_at=datetime('now')
                """,
                (chunk_id, embedding_model, dims, buffer),
            )
            count += 1

        self.conn.commit()
        return count

    @staticmethod
    def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        if not vector_a or not vector_b or len(vector_a) != len(vector_b):
            return 0.0

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for comp_a, comp_b in zip(vector_a, vector_b):
            dot += comp_a * comp_b
            norm_a += comp_a * comp_a
            norm_b += comp_b * comp_b
        denom = math.sqrt(norm_a) * math.sqrt(norm_b)
        if denom <= 0:
            return 0.0
        return dot / denom

    def dense_search(self, query_vector: list[float], embedding_model: str, top_k: int = 8) -> list[RetrievalResult]:
        rows = self.conn.execute(
            """
            SELECT c.chunk_id, c.source_path, c.title, c.section, c.text, e.vector_blob, e.dims
                , c.branch_hint, c.cloud_url, c.page_number
            FROM chunk_embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE e.embedding_model = ?
            """,
            (embedding_model,),
        ).fetchall()

        scored: list[RetrievalResult] = []
        for row in rows:
            vector = array("f")
            vector.frombytes(row["vector_blob"])
            score = self._cosine_similarity(query_vector, list(vector))
            if score <= 0:
                continue
            scored.append(
                RetrievalResult(
                    chunk_id=row["chunk_id"],
                    source_path=row["source_path"],
                    title=row["title"],
                    section=row["section"],
                    text=row["text"],
                    score=score,
                    dense_score=score,
                    source_branch=row["branch_hint"],
                    cloud_url=row["cloud_url"],
                    page_number=row["page_number"],
                    source_url=row["cloud_url"] or row["source_path"],
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        query = query.strip()
        if not query:
            return []

        safe_query = self._safe_fts_query(query)

        def run_fts(fts_query: str) -> list[sqlite3.Row]:
            return self.conn.execute(
                """
                SELECT c.chunk_id, c.source_path, c.title, c.section, c.text,
                       bm25(chunks_fts, 2.0, 1.0, 1.5) as rank,
                       c.branch_hint,
                      c.cloud_url,
                      c.page_number
                FROM chunks_fts
                JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, top_k),
            ).fetchall()

        try:
            rows = run_fts(safe_query)
        except sqlite3.OperationalError:
            tokens = [token for token in safe_query.split() if token]
            fallback_query = " OR ".join(f'"{token}"' for token in tokens[:12])
            if not fallback_query:
                return []
            rows = run_fts(fallback_query)

        results: list[RetrievalResult] = []
        for row in rows:
            score = float(-row["rank"])
            results.append(
                RetrievalResult(
                    chunk_id=row["chunk_id"],
                    source_path=row["source_path"],
                    title=row["title"],
                    section=row["section"],
                    text=row["text"],
                    score=score,
                    source_branch=row["branch_hint"],
                    cloud_url=row["cloud_url"],
                    page_number=row["page_number"],
                    source_url=row["cloud_url"] or row["source_path"],
                )
            )
        return results

    @staticmethod
    def _safe_fts_query(query: str) -> str:
        normalized = query.replace("\n", " ")
        keep = []
        for char in normalized:
            if char.isalnum() or char in {"_", "-", " ", "/"}:
                keep.append(char)
            else:
                keep.append(" ")
        compact = " ".join("".join(keep).split())
        if not compact:
            return ""
        terms = [term for term in compact.split(" ") if term]
        return " ".join(f'"{term}"' for term in terms[:24])

    def get_sync_state(self, connector: str) -> dict[str, dict[str, str | None]]:
        rows = self.conn.execute(
            """
            SELECT external_id, version_id, source_path, cloud_url
            FROM sync_state
            WHERE connector = ?
            """,
            (connector,),
        ).fetchall()
        state: dict[str, dict[str, str | None]] = {}
        for row in rows:
            state[str(row["external_id"])] = {
                "version_id": row["version_id"],
                "source_path": row["source_path"],
                "cloud_url": row["cloud_url"],
            }
        return state

    def upsert_sync_state(
        self,
        *,
        connector: str,
        external_id: str,
        version_id: str | None,
        source_path: str | None,
        cloud_url: str | None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO sync_state (connector, external_id, version_id, source_path, cloud_url, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(connector, external_id) DO UPDATE SET
                version_id=excluded.version_id,
                source_path=excluded.source_path,
                cloud_url=excluded.cloud_url,
                updated_at=datetime('now')
            """,
            (connector, external_id, version_id, source_path, cloud_url),
        )
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        chunk_count = self.conn.execute("SELECT COUNT(*) as value FROM chunks").fetchone()["value"]
        source_count = self.conn.execute("SELECT COUNT(DISTINCT source_path) as value FROM chunks").fetchone()["value"]
        trace_count = self.conn.execute("SELECT COUNT(*) as value FROM query_trace").fetchone()["value"]
        asset_count = self.conn.execute("SELECT COUNT(*) as value FROM assets").fetchone()["value"]
        connected_sources = self.conn.execute(
            "SELECT COUNT(*) as value FROM source_connections WHERE enabled = 1"
        ).fetchone()["value"]
        return {
            "chunk_count": int(chunk_count),
            "source_count": int(source_count),
            "trace_count": int(trace_count),
            "asset_count": int(asset_count),
            "connected_sources": int(connected_sources),
        }

    def connector_breakdown(self) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT connector, COUNT(*) AS chunks, COUNT(DISTINCT source_path) AS sources
            FROM chunks
            GROUP BY connector
            ORDER BY chunks DESC, connector ASC
            """
        ).fetchall()
        return list(rows)

    def upsert_assets(self, assets: list[dict]) -> int:
        if not assets:
            return 0
        count = 0
        for item in assets:
            self.conn.execute(
                """
                INSERT INTO assets (
                    asset_id, source_path, title, asset_type, page_number, file_path,
                    bbox_json, caption_text, ocr_text, figure_id, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(asset_id) DO UPDATE SET
                    source_path=excluded.source_path,
                    title=excluded.title,
                    asset_type=excluded.asset_type,
                    page_number=excluded.page_number,
                    file_path=excluded.file_path,
                    bbox_json=excluded.bbox_json,
                    caption_text=excluded.caption_text,
                    ocr_text=excluded.ocr_text,
                    figure_id=excluded.figure_id,
                    updated_at=datetime('now')
                """,
                (
                    item.get("asset_id"),
                    item.get("source_path"),
                    item.get("title"),
                    item.get("asset_type"),
                    item.get("page_number"),
                    item.get("file_path"),
                    item.get("bbox_json"),
                    item.get("caption_text"),
                    item.get("ocr_text"),
                    item.get("figure_id"),
                ),
            )
            count += 1
        self.conn.commit()
        return count

    def upsert_source_connection(
        self,
        *,
        source_type: str,
        enabled: bool,
        account_label: str | None,
        config_json: str | None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO source_connections (source_type, enabled, account_label, config_json, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(source_type) DO UPDATE SET
                enabled=excluded.enabled,
                account_label=excluded.account_label,
                config_json=excluded.config_json,
                updated_at=datetime('now')
            """,
            (source_type, 1 if enabled else 0, account_label, config_json),
        )
        self.conn.commit()

    def disable_source_connection(self, source_type: str) -> None:
        self.conn.execute(
            """
            INSERT INTO source_connections (source_type, enabled, updated_at)
            VALUES (?, 0, datetime('now'))
            ON CONFLICT(source_type) DO UPDATE SET
                enabled=0,
                updated_at=datetime('now')
            """,
            (source_type,),
        )
        self.conn.commit()

    def list_source_connections(self) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT source_type, enabled, account_label, config_json, updated_at
            FROM source_connections
            ORDER BY source_type ASC
            """
        ).fetchall()
        return list(rows)

    def last_sync_time(self) -> str | None:
        row = self.conn.execute("SELECT MAX(updated_at) AS value FROM sync_state").fetchone()
        if not row:
            return None
        return row["value"]

    def write_trace(
        self,
        *,
        query: str,
        top_k: int,
        provider: str,
        retrieval_mode: str,
        git_branch: str | None,
        cwd: str,
        result_count: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO query_trace (query, top_k, provider, retrieval_mode, git_branch, cwd, result_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (query, top_k, provider, retrieval_mode, git_branch, cwd, result_count),
        )
        self.conn.commit()

    def recent_traces(self, limit: int = 10) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT id, created_at, query, top_k, provider, retrieval_mode, git_branch, cwd, result_count
            FROM query_trace
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return list(rows)

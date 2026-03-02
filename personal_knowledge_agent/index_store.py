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
                updated_at TEXT NOT NULL
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
            """
        )
        self._migrate_schema()
        self.conn.commit()

    def _migrate_schema(self) -> None:
        trace_columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(query_trace)").fetchall()
        }
        if "retrieval_mode" not in trace_columns:
            self.conn.execute("ALTER TABLE query_trace ADD COLUMN retrieval_mode TEXT")

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
                INSERT INTO chunks (chunk_id, source_path, source_type, title, fingerprint, section, text, token_estimate, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    source_path=excluded.source_path,
                    source_type=excluded.source_type,
                    title=excluded.title,
                    fingerprint=excluded.fingerprint,
                    section=excluded.section,
                    text=excluded.text,
                    token_estimate=excluded.token_estimate,
                    updated_at=excluded.updated_at
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
            SELECT c.chunk_id, c.source_path, c.source_type, c.title, c.fingerprint, c.section, c.text, c.token_estimate, c.updated_at
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
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def search(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        query = query.strip()
        if not query:
            return []

        rows = self.conn.execute(
            """
            SELECT c.chunk_id, c.source_path, c.title, c.section, c.text,
                   bm25(chunks_fts, 2.0, 1.0, 1.5) as rank
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, top_k),
        ).fetchall()

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
                )
            )
        return results

    def stats(self) -> dict[str, int]:
        chunk_count = self.conn.execute("SELECT COUNT(*) as value FROM chunks").fetchone()["value"]
        source_count = self.conn.execute("SELECT COUNT(DISTINCT source_path) as value FROM chunks").fetchone()["value"]
        trace_count = self.conn.execute("SELECT COUNT(*) as value FROM query_trace").fetchone()["value"]
        return {
            "chunk_count": int(chunk_count),
            "source_count": int(source_count),
            "trace_count": int(trace_count),
        }

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

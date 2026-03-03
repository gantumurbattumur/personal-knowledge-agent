from __future__ import annotations

import importlib
from pathlib import Path

from .schemas import Chunk, RetrievalResult


class LanceVectorStore:
    def __init__(self, db_path: Path, table_name: str = "chunk_embeddings"):
        self.db_path = db_path.expanduser().resolve()
        self.table_name = table_name
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = None
        self._table = None
        self._enabled = False
        self._reason = "uninitialized"
        self._init()

    def _init(self) -> None:
        try:
            lancedb = importlib.import_module("lancedb")
        except ModuleNotFoundError:
            self._enabled = False
            self._reason = "lancedb is not installed"
            return

        try:
            self._db = lancedb.connect(str(self.db_path))
            self._table = self._db.create_table(
                self.table_name,
                data=[
                    {
                        "chunk_id": "__bootstrap__",
                        "embedding_model": "bootstrap",
                        "vector": [0.0],
                        "source_path": "",
                        "title": "",
                        "section": "",
                        "text": "",
                        "source_branch": None,
                        "cloud_url": None,
                        "page_number": None,
                        "source_url": None,
                    }
                ],
                mode="create_if_not_exists",
            )
            self._table.delete("chunk_id = '__bootstrap__'")
            self._enabled = True
            self._reason = "ok"
        except Exception as exc:
            self._enabled = False
            self._reason = f"failed to initialize LanceDB: {exc}"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def reason(self) -> str:
        return self._reason

    def upsert_embeddings(self, embedding_model: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if not self._enabled or self._table is None or not chunks or not vectors:
            return 0

        rows: list[dict] = []
        for chunk, vector in zip(chunks, vectors):
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "embedding_model": embedding_model,
                    "vector": vector,
                    "source_path": chunk.source_path,
                    "title": chunk.title,
                    "section": chunk.section,
                    "text": chunk.text,
                    "source_branch": chunk.metadata.branch_hint,
                    "cloud_url": chunk.metadata.cloud_url,
                    "page_number": chunk.metadata.page_number,
                    "source_url": chunk.metadata.source_url or chunk.metadata.cloud_url,
                }
            )

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self._table.delete(f"chunk_id IN ({','.join(repr(cid) for cid in chunk_ids)})")
        self._table.add(rows)
        return len(rows)

    def search(self, query_vector: list[float], embedding_model: str, top_k: int = 8) -> list[RetrievalResult]:
        if not self._enabled or self._table is None:
            return []

        try:
            result_table = (
                self._table.search(query_vector)
                .where(f"embedding_model = '{embedding_model}'")
                .limit(top_k)
                .to_arrow()
            )
        except Exception:
            return []

        results: list[RetrievalResult] = []
        for row in result_table.to_pylist():
            distance = float(row.get("_distance", 0.0) or 0.0)
            score = 1.0 / (1.0 + max(distance, 0.0))
            results.append(
                RetrievalResult(
                    chunk_id=str(row.get("chunk_id", "")),
                    source_path=str(row.get("source_path", "")),
                    title=str(row.get("title", "")),
                    section=str(row.get("section", "")),
                    text=str(row.get("text", "")),
                    score=score,
                    dense_score=score,
                    source_branch=row.get("source_branch"),
                    cloud_url=row.get("cloud_url"),
                    page_number=row.get("page_number"),
                    source_url=row.get("source_url") or row.get("cloud_url") or str(row.get("source_path", "")),
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

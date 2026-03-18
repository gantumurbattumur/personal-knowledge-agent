from __future__ import annotations

from personal_knowledge_agent.index_store import IndexStore
from personal_knowledge_agent.schemas import Chunk


def _chunk(chunk_id: str, generation: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_path=f"/tmp/{chunk_id}.md",
        source_type="text",
        title=chunk_id,
        fingerprint=f"fp-{chunk_id}",
        section="root",
        text=f"content-{chunk_id}",
        token_estimate=4,
        updated_at="2026-03-17T00:00:00Z",
        index_generation=generation,
        chunking_profile_id="v1-fixed",
        embedding_profile_id="v1-minilm",
    )


def test_chunks_missing_embeddings_respects_generation_filter(tmp_path):
    store = IndexStore(tmp_path / "index.sqlite3")
    try:
        store.upsert_chunks([_chunk("c1", "g1"), _chunk("c2", "g2")])

        all_missing = store.chunks_missing_embeddings("text-embedding-3-small")
        assert {item.chunk_id for item in all_missing} == {"c1", "c2"}

        g1_missing = store.chunks_missing_embeddings("text-embedding-3-small", index_generation="g1")
        assert [item.chunk_id for item in g1_missing] == ["c1"]

        store.upsert_embeddings("text-embedding-3-small", ["c1"], [[0.1, 0.2, 0.3]])

        g1_after = store.chunks_missing_embeddings("text-embedding-3-small", index_generation="g1")
        assert g1_after == []

        g2_after = store.chunks_missing_embeddings("text-embedding-3-small", index_generation="g2")
        assert [item.chunk_id for item in g2_after] == ["c2"]
    finally:
        store.close()


def test_upsert_chunks_persists_generation_profile_fields(tmp_path):
    store = IndexStore(tmp_path / "index.sqlite3")
    try:
        chunk = _chunk("persist-me", "gen-alpha")
        chunk.chunking_profile_id = "v2-semantic"
        chunk.embedding_profile_id = "v3-openai"
        store.upsert_chunks([chunk])

        row = store.conn.execute(
            """
            SELECT index_generation, chunking_profile_id, embedding_profile_id
            FROM chunks
            WHERE chunk_id = ?
            """,
            ("persist-me",),
        ).fetchone()

        assert row is not None
        assert row["index_generation"] == "gen-alpha"
        assert row["chunking_profile_id"] == "v2-semantic"
        assert row["embedding_profile_id"] == "v3-openai"
    finally:
        store.close()

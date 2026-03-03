from __future__ import annotations

from pathlib import Path

from .embeddings import EmbeddingEngine, RerankerEngine
from .config import Settings
from .context import collect_workflow_context
from .index_store import IndexStore
from .schemas import RetrievalResult, WorkflowContext
from .vector_store import LanceVectorStore


def _fuse_results_rrf(
    lexical: list[RetrievalResult],
    dense: list[RetrievalResult],
    top_k: int,
    rrf_k: int = 60,
    lexical_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[RetrievalResult]:
    by_chunk: dict[str, RetrievalResult] = {}

    for rank, item in enumerate(lexical, start=1):
        existing = by_chunk.get(item.chunk_id)
        if not existing:
            existing = RetrievalResult(
                chunk_id=item.chunk_id,
                source_path=item.source_path,
                title=item.title,
                section=item.section,
                text=item.text,
                score=0.0,
                source_branch=item.source_branch,
                cloud_url=item.cloud_url,
            )
            by_chunk[item.chunk_id] = existing
        existing.lexical_score = item.score
        existing.score += lexical_weight * (1.0 / (rrf_k + rank))

    for rank, item in enumerate(dense, start=1):
        existing = by_chunk.get(item.chunk_id)
        if not existing:
            existing = RetrievalResult(
                chunk_id=item.chunk_id,
                source_path=item.source_path,
                title=item.title,
                section=item.section,
                text=item.text,
                score=0.0,
                source_branch=item.source_branch,
                cloud_url=item.cloud_url,
            )
            by_chunk[item.chunk_id] = existing
        existing.dense_score = item.score
        existing.score += dense_weight * (1.0 / (rrf_k + rank))

    fused = list(by_chunk.values())
    for result in fused:
        result.selected_reason = "hybrid_rrf"
    fused.sort(key=_stable_sort_key)
    return fused[:top_k]


def _stable_sort_key(item: RetrievalResult) -> tuple:
    return (
        -item.score,
        item.source_path,
        item.page_number if item.page_number is not None else 10**9,
        item.section,
        item.chunk_id,
    )


def _apply_context_boost(context: WorkflowContext, results: list[RetrievalResult]) -> None:
    cwd = Path(context.cwd).expanduser().resolve()

    for result in results:
        in_cwd = False
        in_branch = False

        try:
            source = Path(result.source_path).expanduser().resolve()
            in_cwd = source.is_relative_to(cwd)
        except Exception:
            in_cwd = False

        if context.git_branch and result.source_branch:
            in_branch = context.git_branch.strip().lower() == result.source_branch.strip().lower()

        if in_cwd or in_branch:
            result.workflow_boost = 1.5
            result.score *= result.workflow_boost
            result.selected_reason = "workflow_boost"


def _maybe_rerank(
    query: str,
    results: list[RetrievalResult],
    settings: Settings,
    max_rerank: int,
) -> list[RetrievalResult]:
    if not settings.enable_reranker or not results:
        return results

    reranker = RerankerEngine(settings.reranker_model, enabled=True)
    if not reranker.status.enabled:
        return results

    subset = results[:max_rerank]
    scores = reranker.rerank(query, [item.text for item in subset])
    if not scores:
        return results

    for item, rerank_score in zip(subset, scores):
        item.rerank_score = rerank_score
        item.score = (item.score * 0.6) + (rerank_score * 0.4)

    combined = subset + results[max_rerank:]
    for item in subset:
        item.selected_reason = "reranked"
    combined.sort(key=_stable_sort_key)
    return combined


def retrieve_with_context(
    store: IndexStore,
    vector_store: LanceVectorStore | None,
    query: str,
    cwd: Path,
    top_k: int,
    settings: Settings,
) -> tuple[WorkflowContext, list[RetrievalResult], str]:
    context = collect_workflow_context(cwd)
    mode = settings.retrieval_mode

    lexical_results: list[RetrievalResult] = []
    dense_results: list[RetrievalResult] = []

    if mode in {"lexical", "hybrid"}:
        lexical_results = store.search(query, top_k=max(top_k * 3, 20))

    if mode in {"dense", "hybrid"}:
        engine = EmbeddingEngine(settings.embedding_model)
        query_vector = engine.encode_query(query) if engine.status.enabled else None
        if query_vector is not None:
            if vector_store and vector_store.enabled:
                dense_results = vector_store.search(
                    query_vector,
                    embedding_model=settings.embedding_model,
                    top_k=max(top_k * 3, 20),
                )
            else:
                dense_results = store.dense_search(query_vector, embedding_model=settings.embedding_model, top_k=max(top_k * 3, 20))
        elif mode == "dense":
            lexical_results = store.search(query, top_k=max(top_k * 3, 20))
            mode = "lexical"

    if mode == "lexical":
        results = lexical_results[:top_k]
        for item in results:
            item.selected_reason = "lexical"
    elif mode == "dense":
        results = dense_results[:top_k]
        for item in results:
            item.selected_reason = "dense"
    else:
        results = _fuse_results_rrf(lexical_results, dense_results, top_k=top_k)

    _apply_context_boost(context, results)
    results.sort(key=_stable_sort_key)
    results = _maybe_rerank(query, results, settings, max_rerank=min(top_k, 12))
    results.sort(key=_stable_sort_key)
    return context, results[:top_k], mode


def build_extractive_answer(query: str, results: list[RetrievalResult], max_snippets: int = 4) -> str:
    if not results:
        return "No grounded evidence found in the local index. Try `rag index <path>` or adjust your query."

    chosen = results[:max_snippets]
    lines = [f"Question: {query}", "", "Grounded evidence:"]
    for idx, item in enumerate(chosen, start=1):
        snippet = " ".join(item.text.split())
        if len(snippet) > 260:
            snippet = snippet[:260].rstrip() + "…"
        lines.append(f"{idx}. {snippet}")
    return "\n".join(lines)


def compress_results_context(results: list[RetrievalResult], max_chars: int = 2600) -> tuple[str, list[str]]:
    consumed = 0
    snippets: list[str] = []
    used_chunks: list[str] = []
    for item in results:
        snippet = " ".join(item.text.split())
        if len(snippet) > 320:
            snippet = snippet[:320].rstrip() + "…"
        if consumed + len(snippet) > max_chars and snippets:
            break
        snippets.append(snippet)
        used_chunks.append(item.chunk_id)
        consumed += len(snippet)
    return "\n\n".join(snippets), used_chunks


def format_references(results: list[RetrievalResult], max_refs: int = 8) -> list[str]:
    refs: list[str] = []
    for item in results[:max_refs]:
        link = item.source_url or item.cloud_url or item.source_path
        location: list[str] = []
        if item.page_number is not None:
            location.append(f"page {item.page_number}")
        if item.slide_number is not None:
            location.append(f"slide {item.slide_number}")
        if item.section_heading or item.section:
            location.append(item.section_heading or item.section)
        location_text = ", ".join(location) if location else "location n/a"
        refs.append(
            f"{link} | {location_text} | chunk={item.chunk_id} | "
            f"score={item.score:.3f} boost={item.workflow_boost:.2f} "
            f"(lex={item.lexical_score:.3f}, dense={item.dense_score:.3f}, rerank={item.rerank_score:.3f})"
        )
    return refs

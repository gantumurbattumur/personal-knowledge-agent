from __future__ import annotations

from pathlib import Path

from .embeddings import EmbeddingEngine, RerankerEngine
from .config import Settings
from .context import collect_workflow_context
from .index_store import IndexStore
from .schemas import RetrievalResult, WorkflowContext


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v - min_v <= 1e-9:
        return [1.0 for _ in values]
    return [(value - min_v) / (max_v - min_v) for value in values]


def _fuse_results(
    lexical: list[RetrievalResult],
    dense: list[RetrievalResult],
    top_k: int,
    lexical_weight: float = 0.55,
    dense_weight: float = 0.45,
) -> list[RetrievalResult]:
    lexical_norm = _normalize_scores([item.score for item in lexical])
    dense_norm = _normalize_scores([item.score for item in dense])

    by_chunk: dict[str, RetrievalResult] = {}

    for item, score in zip(lexical, lexical_norm):
        existing = by_chunk.get(item.chunk_id)
        if not existing:
            existing = RetrievalResult(
                chunk_id=item.chunk_id,
                source_path=item.source_path,
                title=item.title,
                section=item.section,
                text=item.text,
                score=0.0,
            )
            by_chunk[item.chunk_id] = existing
        existing.lexical_score = score

    for item, score in zip(dense, dense_norm):
        existing = by_chunk.get(item.chunk_id)
        if not existing:
            existing = RetrievalResult(
                chunk_id=item.chunk_id,
                source_path=item.source_path,
                title=item.title,
                section=item.section,
                text=item.text,
                score=0.0,
            )
            by_chunk[item.chunk_id] = existing
        existing.dense_score = score

    fused = list(by_chunk.values())
    for item in fused:
        item.score = (item.lexical_score * lexical_weight) + (item.dense_score * dense_weight)
    fused.sort(key=lambda result: result.score, reverse=True)
    return fused[:top_k]


def _apply_context_boost(context: WorkflowContext, results: list[RetrievalResult]) -> None:
    if context.git_branch:
        branch_terms = [term for term in context.git_branch.replace("/", " ").split() if len(term) > 2]
        if branch_terms:
            boost_set = set(branch_terms)
            for result in results:
                haystack = f"{result.title} {result.section} {result.text[:350]}".lower()
                matched = sum(1 for term in boost_set if term.lower() in haystack)
                if matched:
                    result.score += matched * 0.25


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
    combined.sort(key=lambda item: item.score, reverse=True)
    return combined


def retrieve_with_context(
    store: IndexStore,
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
            dense_results = store.dense_search(query_vector, embedding_model=settings.embedding_model, top_k=max(top_k * 3, 20))
        elif mode == "dense":
            lexical_results = store.search(query, top_k=max(top_k * 3, 20))
            mode = "lexical"

    if mode == "lexical":
        results = lexical_results[:top_k]
    elif mode == "dense":
        results = dense_results[:top_k]
    else:
        results = _fuse_results(lexical_results, dense_results, top_k=top_k)

    _apply_context_boost(context, results)
    results.sort(key=lambda r: r.score, reverse=True)
    results = _maybe_rerank(query, results, settings, max_rerank=min(top_k, 12))
    return context, results[:top_k], mode


def build_extractive_answer(query: str, results: list[RetrievalResult], max_snippets: int = 4) -> str:
    if not results:
        return "No grounded evidence found in the local index. Try `pka index <path>` or adjust your query."

    chosen = results[:max_snippets]
    lines = [f"Question: {query}", "", "Grounded evidence:"]
    for idx, item in enumerate(chosen, start=1):
        snippet = " ".join(item.text.split())
        if len(snippet) > 260:
            snippet = snippet[:260].rstrip() + "…"
        lines.append(f"{idx}. {snippet}")
    return "\n".join(lines)


def format_references(results: list[RetrievalResult], max_refs: int = 8) -> list[str]:
    refs: list[str] = []
    for item in results[:max_refs]:
        refs.append(
            f"{item.source_path} | {item.section} | chunk={item.chunk_id} | "
            f"score={item.score:.3f} (lex={item.lexical_score:.3f}, dense={item.dense_score:.3f}, rerank={item.rerank_score:.3f})"
        )
    return refs

from __future__ import annotations

import re
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


def _query_tokens(query: str) -> list[str]:
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    tokens = [token for token in raw_tokens if len(token) >= 3]
    expanded = set(tokens)
    if "genai" in expanded or ("generative" in expanded and "ai" in expanded):
        expanded.update({"llm", "llms", "model", "models", "transformer", "foundation", "generative"})
    return sorted(expanded)


def _apply_query_alignment(query: str, results: list[RetrievalResult]) -> None:
    tokens = _query_tokens(query)
    if not tokens:
        return

    q_lower = query.lower()
    genai_intent = any(keyword in q_lower for keyword in ["genai", "generative", "llm", "foundation model", "foundation-model"])
    genai_markers = [" ai ", " llm", "generative", "foundation model", "transformer", "language model"]

    for item in results:
        title_text = (item.title or "").lower()
        body_text = (item.text or "").lower()
        overlap = 0
        title_overlap = 0
        for token in tokens:
            in_title = token in title_text
            in_body = token in body_text
            if in_title or in_body:
                overlap += 1
                if in_title:
                    title_overlap += 1

        if overlap == 0:
            item.score *= 0.78
            continue

        if genai_intent:
            combined = f" {title_text} {body_text} "
            if not any(marker in combined for marker in genai_markers):
                item.score *= 0.62

        body_boost = 1.0 + min(overlap, 6) * 0.06
        title_boost = 1.0 + min(title_overlap, 3) * 0.05
        item.score *= body_boost * title_boost
        if title_overlap > 0:
            item.selected_reason = "query_aligned"


def _diversify_by_source(results: list[RetrievalResult], top_k: int, max_per_source: int = 2) -> list[RetrievalResult]:
    if not results:
        return []

    buckets: dict[str, list[RetrievalResult]] = {}
    source_order: list[str] = []
    for item in results:
        key = item.source_path
        if key not in buckets:
            buckets[key] = []
            source_order.append(key)
        buckets[key].append(item)

    source_order.sort(key=lambda key: -buckets[key][0].score if buckets[key] else 0.0)

    picked_per_source: dict[str, int] = {key: 0 for key in source_order}
    chosen: list[RetrievalResult] = []

    while len(chosen) < top_k:
        progressed = False
        for source_key in source_order:
            if len(chosen) >= top_k:
                break
            if picked_per_source[source_key] >= max_per_source:
                continue
            bucket = buckets[source_key]
            if not bucket:
                continue
            chosen.append(bucket.pop(0))
            picked_per_source[source_key] += 1
            progressed = True
        if not progressed:
            break

    if len(chosen) < top_k:
        leftovers: list[RetrievalResult] = []
        for source_key in source_order:
            leftovers.extend(buckets[source_key])
        leftovers.sort(key=_stable_sort_key)
        for item in leftovers:
            chosen.append(item)
            if len(chosen) >= top_k:
                break

    return chosen[:top_k]


def _apply_context_boost(context: WorkflowContext, results: list[RetrievalResult]) -> None:
    cwd = Path(context.cwd).expanduser().resolve()

    for result in results:
        in_cwd = False
        in_branch = False

        try:
            source = Path(result.source_path).expanduser().resolve()
            in_cwd = source.is_relative_to(cwd)
            if "/.pka/sources/" in source.as_posix():
                in_cwd = False
        except Exception:
            in_cwd = False

        if context.git_branch and result.source_branch:
            in_branch = context.git_branch.strip().lower() == result.source_branch.strip().lower()

        if in_cwd or in_branch:
            result.workflow_boost = 1.5
            result.score *= result.workflow_boost
            result.selected_reason = "workflow_boost"


def _should_apply_workflow_boost(query: str) -> bool:
    q = query.lower()
    workflow_markers = (
        "repo",
        "repository",
        "code",
        "function",
        "class",
        "method",
        "file",
        "bug",
        "error",
        "traceback",
        "cli",
        "config",
        "pyproject",
        "readme",
    )
    return any(marker in q for marker in workflow_markers)


def _apply_external_source_preference(settings: Settings, results: list[RetrievalResult]) -> None:
    if not settings.prefer_external_sources:
        return

    project_root = settings.project_root.expanduser().resolve()
    for result in results:
        try:
            source = Path(result.source_path).expanduser().resolve()
        except Exception:
            continue

        source_posix = source.as_posix()
        in_synced_sources = "/.pka/sources/" in source_posix
        in_workspace = False
        try:
            in_workspace = source.is_relative_to(project_root)
        except Exception:
            in_workspace = False

        if in_workspace and not in_synced_sources:
            result.score *= 0.72
            if not result.selected_reason:
                result.selected_reason = "workspace_penalty"


def _order_preferred_sources(settings: Settings, results: list[RetrievalResult]) -> list[RetrievalResult]:
    if not settings.prefer_external_sources or not results:
        return results

    external: list[RetrievalResult] = []
    fallback: list[RetrievalResult] = []
    for result in results:
        source = (result.source_path or "").replace("\\", "/")
        if "/.pka/sources/" in source:
            external.append(result)
        else:
            fallback.append(result)
    return external + fallback


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
    candidate_pool = max(top_k * 6, 36)
    active_generation = settings.active_index_generation

    lexical_results: list[RetrievalResult] = []
    dense_results: list[RetrievalResult] = []

    if mode in {"lexical", "hybrid"}:
        lexical_results = store.search(query, top_k=max(top_k * 3, 20), index_generation=active_generation)

    if mode in {"dense", "hybrid"}:
        engine = EmbeddingEngine(settings.embedding_model, provider=settings.embedding_provider)
        query_vector = engine.encode_query(query) if engine.status.enabled else None
        if query_vector is not None:
            if vector_store and vector_store.enabled:
                dense_results = vector_store.search(
                    query_vector,
                    embedding_model=settings.embedding_model,
                    top_k=max(top_k * 3, 20),
                )
            else:
                dense_results = store.dense_search(query_vector, embedding_model=settings.embedding_model, top_k=max(top_k * 3, 20), index_generation=active_generation)
        elif mode == "dense":
            lexical_results = store.search(query, top_k=max(top_k * 3, 20), index_generation=active_generation)
            mode = "lexical"

    if mode == "lexical":
        results = lexical_results[:candidate_pool]
        for item in results:
            item.selected_reason = "lexical"
    elif mode == "dense":
        results = dense_results[:candidate_pool]
        for item in results:
            item.selected_reason = "dense"
    else:
        results = _fuse_results_rrf(lexical_results, dense_results, top_k=candidate_pool)

    if _should_apply_workflow_boost(query):
        _apply_context_boost(context, results)
    _apply_external_source_preference(settings, results)
    _apply_query_alignment(query, results)
    results.sort(key=_stable_sort_key)
    
    # Rerank larger candidate pool before diversification (reranker always on when enabled)
    rerank_pool = min(len(results), max(top_k * 3, 24))
    results = _maybe_rerank(query, results, settings, max_rerank=rerank_pool)
    results.sort(key=_stable_sort_key)
    results = _order_preferred_sources(settings, results)
    
    results = _diversify_by_source(results, top_k=top_k, max_per_source=2)
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


def build_concise_grounded_answer(question: str, results: list[RetrievalResult], max_points: int = 4) -> str:
    if not results:
        return "Insufficient indexed evidence to answer this question."

    lines = [f"Question: {question}", "", "Grounded summary:"]
    seen = set()
    points = 0

    def best_sentence(text: str) -> str:
        compact = " ".join(text.split())
        if not compact:
            return ""
        candidates = re.split(r"(?<=[.!?])\s+", compact)
        for candidate in candidates:
            cleaned = candidate.strip()
            if len(cleaned) < 60:
                continue
            digit_ratio = sum(ch.isdigit() for ch in cleaned) / max(len(cleaned), 1)
            if digit_ratio > 0.25:
                continue
            if "www." in cleaned.lower():
                continue
            return cleaned
        return compact[:220].rstrip() + ("…" if len(compact) > 220 else "")

    for item in results:
        snippet = best_sentence(item.text)
        if not snippet:
            continue
        normalized = snippet[:180].lower()
        if normalized in seen:
            continue
        seen.add(normalized)

        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "…"

        source_label = item.title or Path(item.source_path).name
        location = f"p.{item.page_number}" if item.page_number is not None else (item.section_heading or item.section or "")
        if location:
            lines.append(f"- {snippet} ({source_label}, {location})")
        else:
            lines.append(f"- {snippet} ({source_label})")
        points += 1
        if points >= max_points:
            break

    if points == 0:
        return "Insufficient indexed evidence to answer this question."
    return "\n".join(lines)


def compress_results_context(results: list[RetrievalResult], max_chars: int = 2600) -> tuple[str, list[str]]:
    consumed = 0
    snippets: list[str] = []
    used_chunks: list[str] = []
    seen_sources: set[str] = set()
    for item in results:
        source_key = f"{item.source_path}:{item.page_number}:{item.section}"
        if source_key in seen_sources and len(snippets) >= 2:
            continue
        seen_sources.add(source_key)
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

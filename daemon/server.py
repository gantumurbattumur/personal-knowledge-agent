"""FastAPI daemon — persistent background service for workspace-aware RAG.

Start with::

    uvicorn daemon.server:app --host 127.0.0.1 --port 8741

or via the CLI::

    rag daemon start
"""
from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from personal_knowledge_agent.config import load_settings
from personal_knowledge_agent.embeddings import EmbeddingEngine, RerankerEngine
from personal_knowledge_agent.index_store import IndexStore
from personal_knowledge_agent.ingest import ingest_path
from personal_knowledge_agent.llm import generate_answer
from personal_knowledge_agent.retrieval import (
    compress_results_context,
    format_references,
    retrieve_with_context,
)
from personal_knowledge_agent.vector_store import LanceVectorStore

from .auth_token import ensure_token
from .registry import WorkspaceRegistry

logger = logging.getLogger("pka.daemon")

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    workspace_hint: Optional[str] = None  # absolute path or "drive://default"
    clipboard_context: Optional[str] = None  # text highlighted by user
    active_file: Optional[str] = None  # full path of open file in editor
    active_code_selection: Optional[str] = None  # highlighted code snippet


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestRequest(BaseModel):
    path: str


class IngestResponse(BaseModel):
    status: str
    chunks_inserted: int = 0
    chunks_skipped: int = 0
    assets_extracted: int = 0
    embeddings_updated: int = 0


class RegisterWorkspaceRequest(BaseModel):
    path: str
    kind: str = "local"  # "local" | "drive"


class HealthResponse(BaseModel):
    status: str
    workspaces_loaded: int


class ErrorResponse(BaseModel):
    error: str
    searched_path: Optional[str] = None
    hint: Optional[str] = None


# ---------------------------------------------------------------------------
# Lifespan — singleton model loading & registry
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup, tear down on shutdown."""
    token = ensure_token()
    app.state.token = token
    logger.info("Daemon token loaded (%s…)", token[:8])

    # Load workspace registry
    registry = WorkspaceRegistry.load()
    app.state.registry = registry
    logger.info("Registry loaded — %d workspace(s)", registry.loaded_count())

    # Pre-load embedding + reranker engines (--warm behaviour by default)
    default_settings = _default_settings()
    try:
        app.state.embedding_engine = EmbeddingEngine(
            default_settings.embedding_model,
            provider=default_settings.embedding_provider,
        )
        logger.info("Embedding engine loaded: %s", default_settings.embedding_model)
    except Exception:
        app.state.embedding_engine = None
        logger.warning("Embedding engine failed to load — will retry per-request")

    try:
        app.state.reranker_engine = RerankerEngine(
            default_settings.reranker_model,
            enabled=default_settings.enable_reranker,
        )
        logger.info("Reranker engine loaded: %s", default_settings.reranker_model)
    except Exception:
        app.state.reranker_engine = None
        logger.warning("Reranker engine failed to load")

    yield

    # Cleanup
    registry.close_all()
    logger.info("Daemon shutdown complete")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PKA Daemon",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow Chrome extension & local clients to reach us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


def _verify_token(request: Request, x_rag_token: Optional[str] = Header(None)) -> None:
    """Validate that the caller passes the correct daemon token."""
    expected = getattr(request.app.state, "token", None)
    if expected and x_rag_token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-RAG-Token")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_settings() -> "Settings":  # type: ignore[name-defined]
    """Fallback settings when no workspace-specific settings are available."""
    try:
        return load_settings()
    except Exception:
        return load_settings(Path.home())


def _route_to_workspace(req: QueryRequest, registry: WorkspaceRegistry):
    """Resolve the target workspace from the request context."""
    # 1. Explicit hint
    if req.workspace_hint:
        ws = registry.get(req.workspace_hint)
        if ws:
            return ws
        # If the hint is an absolute path, try registering on-the-fly
        hint_path = Path(req.workspace_hint)
        if hint_path.is_dir() and (hint_path / ".pka").is_dir():
            return registry.register(str(hint_path), kind="local")

    # 2. Walk up from active_file
    if req.active_file:
        ws = registry.find_nearest(Path(req.active_file))
        if ws:
            return ws

    # 3. Fall back to Drive
    ws = registry.get("drive://default")
    if ws:
        return ws

    return None


def build_prompt(req: QueryRequest, rag_context: str) -> str:
    """Assemble a context-enriched prompt from all client signals."""
    parts: list[str] = []
    if req.clipboard_context:
        parts.append(f"[User's selected text]:\n{req.clipboard_context}")
    if req.active_file:
        parts.append(f"[User's current file]: {req.active_file}")
    if req.active_code_selection:
        parts.append(f"[User's selected code]:\n{req.active_code_selection}")
    if rag_context:
        parts.append(f"[Retrieved context from workspace]:\n{rag_context}")
    parts.append(f"[Question]: {req.query}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    registry: WorkspaceRegistry = request.app.state.registry
    return HealthResponse(status="ok", workspaces_loaded=registry.loaded_count())


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request, x_rag_token: Optional[str] = Header(None)):
    _verify_token(request, x_rag_token)
    registry: WorkspaceRegistry = request.app.state.registry

    ws = _route_to_workspace(req, registry)
    if ws is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_workspace_found",
                "searched_path": req.workspace_hint or req.active_file,
                "hint": "Run `rag init /path/to/your/repo` to index this workspace",
            },
        )

    try:
        settings = ws.settings
        store = ws.store
        vector_store = ws.vector_store

        cwd = Path(ws.path) if ws.kind == "local" else Path.home()

        _ctx, results, _mode = retrieve_with_context(
            store=store,
            vector_store=vector_store,
            query=req.query,
            cwd=cwd,
            top_k=8,
            settings=settings,
        )

        # Compress results into LLM-friendly evidence
        evidence, _used_chunks = compress_results_context(results)

        # Build context-enriched prompt
        prompt = build_prompt(req, evidence)

        # Generate answer via configured LLM
        answer = generate_answer(settings, prompt, evidence)

        # Format source references
        sources = format_references(results, max_refs=5, project_root=settings.project_root)

        return QueryResponse(answer=answer, sources=sources)

    except Exception as exc:
        logger.error("Query failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request, x_rag_token: Optional[str] = Header(None)):
    _verify_token(request, x_rag_token)
    registry: WorkspaceRegistry = request.app.state.registry

    root = Path(req.path).expanduser().resolve()
    if not root.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {root}")

    # Register if not yet known
    ws = registry.get(str(root))
    if ws is None:
        ws = registry.register(str(root), kind="local")

    try:
        settings = ws.settings
        store = ws.store
        vector_store = ws.vector_store

        bundle = ingest_path(root, assets_dir=settings.cache_dir / "assets")
        chunks = bundle.chunks

        inserted, skipped, updated_chunks = store.upsert_chunks(chunks)
        assets_upserted = (
            store.upsert_assets([a.model_dump() for a in bundle.assets])
            if bundle.assets
            else 0
        )

        embedded = 0
        if settings.retrieval_mode in {"dense", "hybrid"}:
            embed_engine = getattr(request.app.state, "embedding_engine", None)
            if embed_engine is None or not embed_engine.status.enabled:
                embed_engine = EmbeddingEngine(settings.embedding_model, provider=settings.embedding_provider)

            if embed_engine.status.enabled:
                missing = store.chunks_missing_embeddings(settings.embedding_model)
                to_embed = updated_chunks + [
                    c for c in missing if c.chunk_id not in {u.chunk_id for u in updated_chunks}
                ]
                if to_embed:
                    vectors = embed_engine.encode([c.text for c in to_embed])
                    embedded = store.upsert_embeddings(
                        settings.embedding_model,
                        [c.chunk_id for c in to_embed],
                        vectors,
                    )
                    if vector_store.enabled:
                        vector_store.upsert_embeddings(settings.embedding_model, to_embed, vectors)

        # Update last_indexed timestamp
        from datetime import datetime, timezone
        ws.last_indexed = datetime.now(timezone.utc).isoformat()
        registry.save()

        return IngestResponse(
            status="ok",
            chunks_inserted=inserted,
            chunks_skipped=skipped,
            assets_extracted=assets_upserted,
            embeddings_updated=embedded,
        )

    except Exception as exc:
        logger.error("Ingest failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/workspaces")
async def list_workspaces(request: Request, x_rag_token: Optional[str] = Header(None)):
    _verify_token(request, x_rag_token)
    registry: WorkspaceRegistry = request.app.state.registry
    return {"workspaces": registry.list_all()}


@app.post("/workspaces")
async def register_workspace(
    req: RegisterWorkspaceRequest,
    request: Request,
    x_rag_token: Optional[str] = Header(None),
):
    _verify_token(request, x_rag_token)
    registry: WorkspaceRegistry = request.app.state.registry
    ws = registry.register(req.path, kind=req.kind)
    return ws.to_dict()

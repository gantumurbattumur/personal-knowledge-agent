from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import CONFIG_PATH, init_default_config, load_settings
from .embeddings import EmbeddingEngine, RerankerEngine
from .index_store import IndexStore
from .ingest import ingest_path
from .llm import generate_answer
from .retrieval import build_extractive_answer, format_references, retrieve_with_context

app = typer.Typer(help="Personal Knowledge Agent (terminal RAG)")
console = Console()


def _store() -> IndexStore:
    settings = load_settings()
    return IndexStore(settings.db_path)


@app.command()
def config(
    action: str = typer.Argument("show", help="show | init"),
    force: bool = typer.Option(False, help="Overwrite existing config when action=init"),
) -> None:
    if action == "init":
        path = init_default_config(force=force)
        console.print(f"Initialized config at: {path}")
        return

    settings = load_settings()
    table = Table(title="PKA Config")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("config_path", str(CONFIG_PATH))
    table.add_row("db_path", str(settings.db_path))
    table.add_row("llm_provider", settings.llm_provider)
    table.add_row("ollama_base_url", settings.ollama_base_url)
    table.add_row("ollama_model", settings.ollama_model)
    table.add_row("openai_model", settings.openai_model)
    table.add_row("retrieval_mode", settings.retrieval_mode)
    table.add_row("embedding_model", settings.embedding_model)
    table.add_row("enable_reranker", str(settings.enable_reranker))
    table.add_row("reranker_model", settings.reranker_model)
    console.print(table)


@app.command()
def index(
    path: str = typer.Argument("."),
) -> None:
    settings = load_settings()
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise typer.BadParameter(f"Path does not exist: {root}")

    console.print(f"Indexing: {root}")
    chunks = ingest_path(root)
    with _store() as store:
        inserted, skipped, updated_chunks = store.upsert_chunks(chunks)

        embedded = 0
        if settings.retrieval_mode in {"dense", "hybrid"}:
            embed_engine = EmbeddingEngine(settings.embedding_model)
            if embed_engine.status.enabled:
                missing_chunks = store.chunks_missing_embeddings(settings.embedding_model)
                to_embed = updated_chunks + [item for item in missing_chunks if item.chunk_id not in {chunk.chunk_id for chunk in updated_chunks}]

                if to_embed:
                    vectors = embed_engine.encode([chunk.text for chunk in to_embed])
                    embedded = store.upsert_embeddings(
                        settings.embedding_model,
                        [chunk.chunk_id for chunk in to_embed],
                        vectors,
                    )
            else:
                console.print(f"Embedding disabled: {embed_engine.status.reason}")

        stats = store.stats()

    console.print(f"Indexed chunks: inserted={inserted}, skipped={skipped}")
    if settings.retrieval_mode in {"dense", "hybrid"}:
        console.print(f"Embeddings updated: {embedded} (model={settings.embedding_model})")
    console.print(f"Total chunks={stats['chunk_count']}, sources={stats['source_count']}")


@app.command()
def search(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(8, min=1, max=30),
    mode: str | None = typer.Option(None, help="Override retrieval mode: lexical | dense | hybrid"),
) -> None:
    settings = load_settings()
    if mode:
        settings.retrieval_mode = mode.lower().strip()

    context_root = Path(".").resolve()
    with _store() as store:
        _, results, used_mode = retrieve_with_context(store, query, context_root, top_k, settings)

    if not results:
        console.print("No results.")
        return

    table = Table(title=f"Search Results ({len(results)}) mode={used_mode}")
    table.add_column("Score", justify="right")
    table.add_column("Lex", justify="right")
    table.add_column("Dense", justify="right")
    table.add_column("Source")
    table.add_column("Section")
    table.add_column("Snippet")
    for item in results:
        snippet = " ".join(item.text.split())
        if len(snippet) > 120:
            snippet = snippet[:120].rstrip() + "…"
        table.add_row(
            f"{item.score:.3f}",
            f"{item.lexical_score:.3f}",
            f"{item.dense_score:.3f}",
            item.source_path,
            item.section,
            snippet,
        )
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(...),
    top_k: int = typer.Option(8, min=1, max=30),
    cwd: str = typer.Option(".", help="Workflow context directory"),
    mode: str | None = typer.Option(None, help="Override retrieval mode: lexical | dense | hybrid"),
) -> None:
    settings = load_settings()
    if mode:
        settings.retrieval_mode = mode.lower().strip()

    context_root = Path(cwd).expanduser().resolve()

    with _store() as store:
        workflow_context, results, used_mode = retrieve_with_context(store, question, context_root, top_k, settings)
        evidence = build_extractive_answer(question, results)
        response = generate_answer(settings, question, evidence)

        store.write_trace(
            query=question,
            top_k=top_k,
            provider=settings.llm_provider,
            retrieval_mode=used_mode,
            git_branch=workflow_context.git_branch,
            cwd=workflow_context.cwd,
            result_count=len(results),
        )

    console.rule("Answer")
    console.print(response)
    console.rule("References")
    refs = format_references(results)
    if refs:
        for line in refs:
            console.print(f"- {line}")
    else:
        console.print("- No references")

    console.rule("Workflow Context")
    console.print(f"retrieval_mode: {used_mode}")
    console.print(f"cwd: {workflow_context.cwd}")
    console.print(f"git_branch: {workflow_context.git_branch or 'N/A'}")
    if workflow_context.recent_files:
        console.print("recent_files:")
        for item in workflow_context.recent_files[:8]:
            console.print(f"  - {item}")


@app.command()
def trace(limit: int = typer.Option(10, min=1, max=100)) -> None:
    with _store() as store:
        rows = store.recent_traces(limit=limit)

    table = Table(title=f"Recent Query Trace ({len(rows)})")
    table.add_column("When")
    table.add_column("Provider")
    table.add_column("Mode")
    table.add_column("Branch")
    table.add_column("Results", justify="right")
    table.add_column("Query")
    for row in rows:
        table.add_row(
            str(row["created_at"]),
            str(row["provider"]),
            str(row["retrieval_mode"] or "N/A"),
            str(row["git_branch"] or "N/A"),
            str(row["result_count"]),
            str(row["query"]),
        )
    console.print(table)


@app.command()
def doctor() -> None:
    settings = load_settings()
    status_table = Table(title="PKA Doctor")
    status_table.add_column("Check")
    status_table.add_column("Result")

    status_table.add_row("config", "ok" if CONFIG_PATH.exists() else "missing")
    status_table.add_row("database", str(settings.db_path))
    status_table.add_row("llm_provider", settings.llm_provider)
    status_table.add_row("retrieval_mode", settings.retrieval_mode)
    status_table.add_row("embedding_model", settings.embedding_model)

    embed_status = EmbeddingEngine(settings.embedding_model).status
    status_table.add_row("embedding_engine", "ok" if embed_status.enabled else f"disabled: {embed_status.reason}")

    reranker_status = RerankerEngine(settings.reranker_model, enabled=settings.enable_reranker).status
    status_table.add_row("reranker", "ok" if reranker_status.enabled else f"disabled: {reranker_status.reason}")

    try:
        with _store() as store:
            stats = store.stats()
        status_table.add_row("chunks", str(stats["chunk_count"]))
        status_table.add_row("sources", str(stats["source_count"]))
    except Exception as exc:
        status_table.add_row("index", f"error: {exc}")

    console.print(status_table)


if __name__ == "__main__":
    app()

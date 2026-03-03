from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .auth import auth_google_drive, auth_notion, load_token, notion_authorize_url
from .config import init_default_config, load_settings, resolve_app_dir
from .context import collect_workflow_context
from .drive import sync_google_drive_incremental
from .embeddings import EmbeddingEngine, RerankerEngine
from .index_store import IndexStore
from .ingest import ingest_path
from .llm import generate_answer
from .notion import sync_notion_incremental
from .retrieval import build_extractive_answer, compress_results_context, format_references, retrieve_with_context
from .vector_store import LanceVectorStore

app = typer.Typer(help="Workflow-aware personal RAG assistant")
sources_app = typer.Typer(help="Manage RAG sources")
rag_app = typer.Typer(help="RAG indexing and status")
app.add_typer(sources_app, name="sources")
app.add_typer(rag_app, name="rag")
console = Console()


def _store() -> IndexStore:
    settings = load_settings()
    return IndexStore(settings.db_path)


def _vector_store() -> LanceVectorStore:
    settings = load_settings()
    return LanceVectorStore(settings.vectorstore_path)


def _rewrite_query(question: str, cwd: Path) -> str:
    context = collect_workflow_context(cwd)
    parts = [question.strip()]
    if context.git_branch:
        parts.append(f"active_branch:{context.git_branch}")
    parts.append(f"cwd:{context.cwd}")
    return "\n".join(parts)


def _token_status(service: str) -> str:
    try:
        return "ok" if load_token(service) else "missing"
    except Exception:
        return "disabled (keyring missing)"


def _llm_key_status(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in {"none", ""}:
        return "n/a"
    if normalized in {"local", "ollama"}:
        return "ok"
    if normalized == "openai":
        return "ok" if os.getenv("OPENAI_API_KEY", "").strip() else "missing key"
    if normalized == "anthropic":
        return "ok" if os.getenv("ANTHROPIC_API_KEY", "").strip() else "missing key"
    return "unknown provider"


def _ensure_gitignore_entries(project_root: Path) -> None:
    gitignore_path = project_root / ".gitignore"
    existing = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
    entries = [".env", ".pka/cache/", ".pka/logs/", ".pka/vectorstore/", ".pka/index.sqlite3"]
    missing = [entry for entry in entries if entry not in existing]
    if not missing:
        return

    with gitignore_path.open("a", encoding="utf-8") as handle:
        if existing and not existing.endswith("\n"):
            handle.write("\n")
        handle.write("\n# Personal Knowledge Agent\n")
        for entry in missing:
            handle.write(f"{entry}\n")


def _connected_services(store: IndexStore) -> dict[str, dict]:
    mapping: dict[str, dict] = {
        "local": {"connected": False, "account_label": "-", "paths": []},
        "google_drive": {"connected": False, "account_label": "-", "paths": []},
        "notion": {"connected": False, "account_label": "-", "paths": []},
        "dropbox": {"connected": False, "account_label": "-", "paths": []},
    }
    for row in store.list_source_connections():
        source_type = str(row["source_type"])
        enabled = bool(row["enabled"])
        account_label = row["account_label"]
        config_raw = row["config_json"]
        details = mapping.setdefault(source_type, {"connected": False, "account_label": "-", "paths": []})
        details["connected"] = enabled
        if account_label:
            details["account_label"] = account_label
        if config_raw:
            try:
                parsed = json.loads(str(config_raw))
            except Exception:
                parsed = {}
            path_value = parsed.get("path") or parsed.get("destination")
            if isinstance(path_value, str) and path_value:
                details["paths"].append(path_value)
    return mapping


@app.command()
def init(force: bool = typer.Option(False, help="Overwrite local .pka/config.toml if it exists")) -> None:
    settings = load_settings(Path.cwd())
    project_root = settings.project_root
    app_dir = resolve_app_dir(project_root)
    app_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_path.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    config_path = init_default_config(force=force, start=project_root)
    env_path = project_root / ".env"
    if force or not env_path.exists():
        env_path.write_text(
            "# Personal Knowledge Agent (.env in project root)\n"
            "PKA_LLM_PROVIDER=none\n"
            "OPENAI_API_KEY=\n"
            "ANTHROPIC_API_KEY=\n"
            "OLLAMA_BASE_URL=http://localhost:11434\n"
            "OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M\n"
            "GOOGLE_DRIVE_FOLDER_ID=\n"
            "GOOGLE_DRIVE_ACCOUNT_LABEL=\n"
            "GOOGLE_DRIVE_DEST=.pka/sources/google-drive\n"
            "NOTION_ROOT_PAGE_ID=\n"
            "NOTION_DEST=.pka/sources/notion\n",
            encoding="utf-8",
        )

    _ensure_gitignore_entries(project_root)

    console.print(f"Initialized local workspace at: {app_dir}")
    console.print(f"Config: {config_path}")
    console.print(f"Env: {env_path}")


@app.command()
def config(
    action: str = typer.Argument("show", help="show | init"),
    force: bool = typer.Option(False, help="Overwrite existing config when action=init"),
) -> None:
    if action == "init":
        path = init_default_config(force=force, start=Path.cwd())
        console.print(f"Initialized config at: {path}")
        return

    settings = load_settings()
    table = Table(title="PKA Config")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("config_path", str(settings.config_path))
    table.add_row("project_root", str(settings.project_root))
    table.add_row("app_dir", str(settings.app_dir))
    table.add_row("db_path", str(settings.db_path))
    table.add_row("vectorstore_path", str(settings.vectorstore_path))
    table.add_row("cache_dir", str(settings.cache_dir))
    table.add_row("logs_dir", str(settings.logs_dir))
    table.add_row("llm_provider", settings.llm_provider)
    table.add_row("ollama_base_url", settings.ollama_base_url)
    table.add_row("ollama_model", settings.ollama_model)
    table.add_row("openai_model", settings.openai_model)
    table.add_row("retrieval_mode", settings.retrieval_mode)
    table.add_row("embedding_model", settings.embedding_model)
    table.add_row("enable_reranker", str(settings.enable_reranker))
    table.add_row("reranker_model", settings.reranker_model)
    console.print(table)


@rag_app.command("build")
def rag_build(
    path: str = typer.Argument("."),
) -> None:
    settings = load_settings()
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise typer.BadParameter(f"Path does not exist: {root}")

    console.print(f"Indexing: {root}")
    bundle = ingest_path(root, assets_dir=settings.cache_dir / "assets")
    chunks = bundle.chunks
    with _store() as store:
        vector_store = _vector_store()
        inserted, skipped, updated_chunks = store.upsert_chunks(chunks)
        assets_upserted = store.upsert_assets([asset.model_dump() for asset in bundle.assets]) if bundle.assets else 0

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
                    if vector_store.enabled:
                        vector_store.upsert_embeddings(settings.embedding_model, to_embed, vectors)
            else:
                console.print(f"Embedding disabled: {embed_engine.status.reason}")

        stats = store.stats()

    console.print(f"Indexed chunks: inserted={inserted}, skipped={skipped}")
    console.print(f"Assets extracted: {assets_upserted}")
    if settings.retrieval_mode in {"dense", "hybrid"}:
        console.print(f"Embeddings updated: {embedded} (model={settings.embedding_model})")
        if not vector_store.enabled:
            console.print(f"LanceDB disabled: {vector_store.reason}. Using SQLite dense fallback.")
    console.print(f"Total chunks={stats['chunk_count']}, sources={stats['source_count']}")


@app.command("index")
def index_alias(path: str = typer.Argument(".")) -> None:
    rag_build(path)


@app.command("build")
def build_alias(path: str = typer.Argument(".")) -> None:
    rag_build(path)


@app.command("auth")
def auth(
    service: str = typer.Argument(..., help="google-drive | notion"),
    client_secret_file: str = typer.Option("client_secret.json", help="OAuth client secret JSON for google-drive"),
    client_id: str = typer.Option("", help="Notion OAuth client id"),
    client_secret: str = typer.Option("", help="Notion OAuth client secret"),
    redirect_uri: str = typer.Option("http://localhost:8765/callback", help="Notion OAuth redirect URI"),
    code: str = typer.Option("", help="Notion authorization code"),
) -> None:
    normalized = service.strip().lower()
    if normalized == "google-drive":
        result = auth_google_drive(client_secret_file=client_secret_file)
        if result.ok:
            console.print(result.message)
            return
        raise typer.BadParameter(result.message)

    if normalized == "notion":
        if not code:
            if not client_id:
                raise typer.BadParameter("--client-id is required for notion auth")
            url = notion_authorize_url(client_id=client_id, redirect_uri=redirect_uri)
            console.print("Open this URL, authorize the app, then run again with --code:")
            console.print(url)
            return
        if not client_id or not client_secret:
            raise typer.BadParameter("--client-id and --client-secret are required with --code")
        result = auth_notion(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, code=code)
        if result.ok:
            console.print(result.message)
            return
        raise typer.BadParameter(result.message)

    raise typer.BadParameter("Unsupported service. Use: google-drive | notion")


@app.command()
def sync() -> None:
    settings = load_settings()
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    destination = Path(os.getenv("GOOGLE_DRIVE_DEST", str(settings.app_dir / "sources/google-drive")))
    account_label = os.getenv("GOOGLE_DRIVE_ACCOUNT_LABEL", "").strip()
    notion_root_page_id = os.getenv("NOTION_ROOT_PAGE_ID", "").strip()
    notion_destination = Path(os.getenv("NOTION_DEST", str(settings.app_dir / "sources/notion")))

    with _store() as store:
        for connection in store.list_source_connections():
            source_type = str(connection["source_type"])
            if not bool(connection["enabled"]):
                continue
            config_raw = connection["config_json"]
            if not config_raw:
                continue
            try:
                config_data = json.loads(str(config_raw))
            except Exception:
                config_data = {}

            if source_type == "google_drive":
                folder_id = str(config_data.get("folder_id") or folder_id).strip()
                destination = Path(str(config_data.get("destination") or destination))
                account_label = str(config_data.get("account_label") or account_label).strip()
            if source_type == "notion":
                notion_root_page_id = str(config_data.get("root_page_id") or notion_root_page_id).strip()
                notion_destination = Path(str(config_data.get("destination") or notion_destination))

    drive_summary = "skipped"
    notion_summary = "skipped"
    sync_paths: list[Path] = []
    if folder_id:
        token_payload = load_token("google-drive")
        if not token_payload:
            raise typer.BadParameter("Google Drive is not authenticated. Run: rag auth google-drive")
        with _store() as store:
            store.upsert_source_connection(
                source_type="google_drive",
                enabled=True,
                account_label=account_label or None,
                config_json=json.dumps({"folder_id": folder_id, "destination": str(destination), "account_label": account_label}),
            )
            result = sync_google_drive_incremental(
                folder_id=folder_id,
                destination=destination,
                token_payload=token_payload,
                store=store,
            )
        if not result.ok:
            raise typer.BadParameter(result.message)
        drive_summary = f"downloaded={result.downloaded}, unchanged={result.unchanged}, destination={result.destination}"
        if result.downloaded > 0:
            sync_paths.append(result.destination)

    if notion_root_page_id:
        token_payload = load_token("notion")
        if not token_payload:
            raise typer.BadParameter("Notion is not authenticated. Run: rag auth notion")
        with _store() as store:
            store.upsert_source_connection(
                source_type="notion",
                enabled=True,
                account_label="notion",
                config_json=json.dumps({"root_page_id": notion_root_page_id, "destination": str(notion_destination)}),
            )
            result = sync_notion_incremental(
                root_page_id=notion_root_page_id,
                destination=notion_destination,
                token_payload=token_payload,
                store=store,
            )
        if not result.ok:
            raise typer.BadParameter(result.message)
        notion_summary = f"downloaded={result.downloaded}, unchanged={result.unchanged}, destination={result.destination}"
        if result.downloaded > 0:
            sync_paths.append(result.destination)

    for sync_path in sync_paths:
        rag_build(path=str(sync_path))

    table = Table(title="RAG Sync")
    table.add_column("Connector")
    table.add_column("Status")
    table.add_row("google-drive", drive_summary)
    table.add_row("notion", notion_summary)
    console.print(table)


@sources_app.command("list")
def sources_list() -> None:
    with _store() as store:
        services = _connected_services(store)

    table = Table(title="Connected Services")
    table.add_column("Service")
    table.add_column("Connected")
    table.add_column("Account")
    table.add_column("Paths")
    for service in ["local", "google_drive", "notion", "dropbox"]:
        entry = services.get(service, {"connected": False, "account_label": "-", "paths": []})
        table.add_row(
            service,
            "yes" if entry["connected"] else "no",
            str(entry.get("account_label") or "-"),
            ", ".join(entry.get("paths", [])) or "-",
        )
    console.print(table)


@sources_app.command("connect")
def sources_connect(
    source_type: str = typer.Argument(..., help="google_drive | local | notion"),
    path: str = typer.Option(".", help="Path for local source"),
    folder_id: str = typer.Option("", help="Google Drive folder id"),
    destination: str = typer.Option("", help="Google Drive destination path"),
    account_label: str = typer.Option("", help="Connector account label"),
    root_page_id: str = typer.Option("", help="Notion root page id"),
) -> None:
    normalized = source_type.strip().lower()
    with _store() as store:
        if normalized == "local":
            resolved = str(Path(path).expanduser().resolve())
            store.upsert_source_connection(
                source_type="local",
                enabled=True,
                account_label="local",
                config_json=json.dumps({"path": resolved}),
            )
            console.print(f"Connected local source: {resolved}")
            return

        if normalized == "google_drive":
            drive_folder = (folder_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")).strip()
            drive_dest = destination or os.getenv("GOOGLE_DRIVE_DEST", str(load_settings().app_dir / "sources/google-drive"))
            if not drive_folder:
                raise typer.BadParameter("Missing folder id. Use --folder-id or set GOOGLE_DRIVE_FOLDER_ID in .env")
            store.upsert_source_connection(
                source_type="google_drive",
                enabled=True,
                account_label=account_label or os.getenv("GOOGLE_DRIVE_ACCOUNT_LABEL", "") or None,
                config_json=json.dumps(
                    {
                        "folder_id": drive_folder,
                        "destination": str(Path(drive_dest).expanduser().resolve()),
                        "account_label": account_label or os.getenv("GOOGLE_DRIVE_ACCOUNT_LABEL", ""),
                    }
                ),
            )
            console.print("Connected source: google_drive")
            return

        if normalized == "notion":
            notion_root = (root_page_id or os.getenv("NOTION_ROOT_PAGE_ID", "")).strip()
            notion_dest = os.getenv("NOTION_DEST", str(load_settings().app_dir / "sources/notion"))
            if not notion_root:
                raise typer.BadParameter("Missing root page id. Use --root-page-id or set NOTION_ROOT_PAGE_ID in .env")
            store.upsert_source_connection(
                source_type="notion",
                enabled=True,
                account_label=account_label or "notion",
                config_json=json.dumps({"root_page_id": notion_root, "destination": str(Path(notion_dest).expanduser().resolve())}),
            )
            console.print("Connected source: notion")
            return

    raise typer.BadParameter("Unsupported source type. Use: local | google_drive | notion")


@sources_app.command("disconnect")
def sources_disconnect(source_type: str = typer.Argument(..., help="Source to disable")) -> None:
    with _store() as store:
        store.disable_source_connection(source_type.strip().lower())
    console.print(f"Disconnected source: {source_type}")


@rag_app.command("status")
def rag_status() -> None:
    with _store() as store:
        stats = store.stats()
        breakdown = store.connector_breakdown()
        last_sync = store.last_sync_time() or "never"

    summary = Table(title="RAG Index Status")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("chunks", str(stats["chunk_count"]))
    summary.add_row("assets", str(stats["asset_count"]))
    summary.add_row("sources", str(stats["source_count"]))
    summary.add_row("connected_services", str(stats["connected_sources"]))
    summary.add_row("last_sync", str(last_sync))
    console.print(summary)

    by_source = Table(title="Connected Sources Breakdown")
    by_source.add_column("Connector")
    by_source.add_column("Chunks", justify="right")
    by_source.add_column("Sources", justify="right")
    if breakdown:
        for row in breakdown:
            by_source.add_row(str(row["connector"]), str(row["chunks"]), str(row["sources"]))
    else:
        by_source.add_row("-", "0", "0")
    console.print(by_source)


@app.command("status")
def status_alias() -> None:
    rag_status()


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
        vector_store = _vector_store()
        _, results, used_mode = retrieve_with_context(store, vector_store, query, context_root, top_k, settings)

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
            item.section_heading or item.section,
            snippet,
        )
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(...),
    top_k: int = typer.Option(8, min=1, max=30),
    cwd: str = typer.Option(".", help="Workflow context directory"),
    mode: str | None = typer.Option(None, help="Override retrieval mode: lexical | dense | hybrid"),
    debug: bool = typer.Option(False, "--debug", help="Show retrieval trace"),
    verbose: bool = typer.Option(False, "--verbose", help="Detailed grounded output"),
    refs: bool = typer.Option(False, "--refs", help="References-only mode"),
    show_rewrite: bool = typer.Option(False, help="Show rewritten query"),
) -> None:
    settings = load_settings()
    if mode:
        settings.retrieval_mode = mode.lower().strip()

    context_root = Path(cwd).expanduser().resolve()

    rewritten_query = _rewrite_query(question, context_root)

    with _store() as store:
        vector_store = _vector_store()
        workflow_context, results, used_mode = retrieve_with_context(store, vector_store, rewritten_query, context_root, top_k, settings)
        compressed_context, used_chunks = compress_results_context(results)
        evidence = build_extractive_answer(question, results)
        response = generate_answer(settings, question, evidence if not compressed_context else compressed_context)

        store.write_trace(
            query=question,
            top_k=top_k,
            provider=settings.llm_provider,
            retrieval_mode=used_mode,
            git_branch=workflow_context.git_branch,
            cwd=workflow_context.cwd,
            result_count=len(results),
        )

    refs_output = format_references(results)

    if refs:
        console.rule("References")
        if refs_output:
            for line in refs_output:
                console.print(f"- {line}")
        else:
            console.print("- No references")
        return

    if debug:
        steps = Table(title="Retrieval Steps")
        steps.add_column("Step")
        steps.add_column("Details")
        steps.add_row("query", question)
        steps.add_row("rewritten", rewritten_query)
        steps.add_row("mode", used_mode)
        steps.add_row("pipeline", "hybrid search -> rerank -> context compression")
        steps.add_row("used_chunks", ", ".join(used_chunks) if used_chunks else "none")
        console.print(steps)

        topk = Table(title=f"Top-K Retrieved ({len(results)})")
        topk.add_column("Rank", justify="right")
        topk.add_column("Score", justify="right")
        topk.add_column("Why")
        topk.add_column("Source")
        topk.add_column("Location")
        for idx, item in enumerate(results, start=1):
            location = []
            if item.page_number is not None:
                location.append(f"page {item.page_number}")
            if item.section_heading or item.section:
                location.append(item.section_heading or item.section)
            topk.add_row(
                str(idx),
                f"{item.score:.3f}",
                item.selected_reason or "score",
                item.source_path,
                ", ".join(location) or "n/a",
            )
        console.print(topk)

    console.rule("Answer")
    console.print(response)

    console.rule("References")
    if refs_output:
        for line in refs_output:
            console.print(f"- {line}")
    else:
        console.print("- No references")

    if verbose or show_rewrite:
        console.rule("Workflow Context")
        if show_rewrite:
            console.print(f"rewritten_query: {rewritten_query}")
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

    status_table.add_row("config", "ok" if settings.config_path.exists() else "missing")
    status_table.add_row("project_root", str(settings.project_root))
    status_table.add_row("project_app_dir", str(settings.app_dir))
    status_table.add_row("database", str(settings.db_path))
    status_table.add_row("vector_index", str(settings.vectorstore_path))
    status_table.add_row("llm_provider", f"{settings.llm_provider} ({_llm_key_status(settings.llm_provider)})")
    status_table.add_row("retrieval_mode", settings.retrieval_mode)
    status_table.add_row("embedding_model", settings.embedding_model)

    embed_status = EmbeddingEngine(settings.embedding_model).status
    status_table.add_row("embedding_engine", "ok" if embed_status.enabled else f"disabled: {embed_status.reason}")
    lancedb_status = _vector_store()
    status_table.add_row("lancedb_engine", "ok" if lancedb_status.enabled else f"disabled: {lancedb_status.reason}")

    reranker_status = RerankerEngine(settings.reranker_model, enabled=settings.enable_reranker).status
    status_table.add_row("reranker", "ok" if reranker_status.enabled else f"disabled: {reranker_status.reason}")
    status_table.add_row("auth_google_drive", _token_status("google-drive"))
    status_table.add_row("auth_notion", _token_status("notion"))

    try:
        with _store() as store:
            stats = store.stats()
        status_table.add_row("chunks", str(stats["chunk_count"]))
        status_table.add_row("sources", str(stats["source_count"]))
        status_table.add_row("assets", str(stats["asset_count"]))
    except Exception as exc:
        status_table.add_row("index", f"error: {exc}")

    console.print(status_table)


@app.command("info")
def info() -> None:
    settings = load_settings()
    with _store() as store:
        stats = store.stats()
        last_sync = store.last_sync_time() or "never"
        services = _connected_services(store)

    console.print(
        Panel.fit(
            f"version: 0.1.0\npython: {sys.version.split()[0]}\nplatform: {platform.platform()}\nproject_root: {settings.project_root}",
            title="RAG Info",
        )
    )

    usage = Table(title="Unified Command Usage")
    usage.add_column("Command")
    usage.add_column("Description")
    usage.add_row("rag init", "Initialize .env, .gitignore, and .pka/ skeleton")
    usage.add_row("rag sources connect <type>", "Connect local/google_drive/notion source")
    usage.add_row("rag sync", "Sync connected cloud sources")
    usage.add_row("rag build [path]", "Build/update local index")
    usage.add_row("rag status", "Show index stats and source breakdown")
    usage.add_row("rag ask \"question\" [--debug]", "Grounded answer with citations")
    usage.add_row("rag search \"query\"", "Retrieval-only lookup")
    usage.add_row("rag doctor", "System health and configuration checks")
    console.print(usage)

    connected = Table(title="Connected Services")
    connected.add_column("Service")
    connected.add_column("Connected")
    connected.add_column("Account")
    connected.add_column("Details")
    local_paths = services.get("local", {}).get("paths", [])
    connected.add_row("Local folders", "yes" if services.get("local", {}).get("connected") else "no", "local", f"count={len(local_paths)} paths={', '.join(local_paths) or '-'}")
    connected.add_row(
        "Google Drive",
        "yes" if services.get("google_drive", {}).get("connected") else "no",
        str(services.get("google_drive", {}).get("account_label") or "-"),
        ", ".join(services.get("google_drive", {}).get("paths", [])) or "-",
    )
    connected.add_row("Notion", "yes" if services.get("notion", {}).get("connected") else "no", str(services.get("notion", {}).get("account_label") or "-"), "placeholder")
    connected.add_row("Dropbox", "yes" if services.get("dropbox", {}).get("connected") else "no", str(services.get("dropbox", {}).get("account_label") or "-"), "placeholder")
    console.print(connected)

    index_summary = Table(title="Indexing Summary")
    index_summary.add_column("Metric")
    index_summary.add_column("Value")
    index_summary.add_row("chunks", str(stats["chunk_count"]))
    index_summary.add_row("sources", str(stats["source_count"]))
    index_summary.add_row("last_sync", str(last_sync))
    console.print(index_summary)


if __name__ == "__main__":
    app()

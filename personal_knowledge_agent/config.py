from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import importlib

    tomllib = importlib.import_module("tomli")


LEGACY_APP_DIR = Path.home() / ".pka"
LEGACY_CONFIG_PATH = LEGACY_APP_DIR / "config.toml"
APP_DIR_NAME = ".pka"


@dataclass(slots=True)
class Settings:
    project_root: Path
    app_dir: Path
    config_path: Path
    db_path: Path
    vectorstore_path: Path
    cache_dir: Path
    logs_dir: Path
    llm_provider: str = "none"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-latest"
    retrieval_mode: str = "hybrid"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @property
    def lancedb_path(self) -> Path:
        return self.vectorstore_path


def _run(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(command, cwd=str(cwd), check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def resolve_project_root(start: Path | None = None) -> Path:
    override = os.getenv("PKA_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    cwd = (start or Path.cwd()).expanduser().resolve()
    git_root = _run(["git", "rev-parse", "--show-toplevel"], cwd)
    if git_root:
        return Path(git_root).expanduser().resolve()
    return cwd


def resolve_app_dir(start: Path | None = None) -> Path:
    return resolve_project_root(start) / APP_DIR_NAME


def resolve_config_path(start: Path | None = None) -> Path:
    return resolve_app_dir(start) / "config.toml"


def resolve_default_db_path(start: Path | None = None) -> Path:
    return resolve_app_dir(start) / "index.sqlite3"


def resolve_default_lancedb_path(start: Path | None = None) -> Path:
    return resolve_app_dir(start) / "vectorstore"


def resolve_default_cache_dir(start: Path | None = None) -> Path:
    return resolve_app_dir(start) / "cache"


def resolve_default_logs_dir(start: Path | None = None) -> Path:
    return resolve_app_dir(start) / "logs"


def ensure_app_dir(start: Path | None = None) -> Path:
    app_dir = resolve_app_dir(start)
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "vectorstore").mkdir(parents=True, exist_ok=True)
    (app_dir / "cache").mkdir(parents=True, exist_ok=True)
    (app_dir / "logs").mkdir(parents=True, exist_ok=True)
    return app_dir


CONFIG_PATH = resolve_config_path()


def init_default_config(force: bool = False, start: Path | None = None) -> Path:
    app_dir = ensure_app_dir(start)
    config_path = app_dir / "config.toml"
    if config_path.exists() and not force:
        return config_path

    config_text = """[storage]
db_path = ".pka/index.sqlite3"
vectorstore_path = ".pka/vectorstore"
cache_dir = ".pka/cache"
logs_dir = ".pka/logs"

[llm]
provider = "none"  # none | local | openai | anthropic
ollama_base_url = "http://localhost:11434"
ollama_model = "llama3.1:8b-instruct-q4_K_M"
openai_model = "gpt-4o-mini"
anthropic_model = "claude-3-5-sonnet-latest"

[retrieval]
mode = "hybrid"  # lexical | dense | hybrid
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
enable_reranker = false
reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
"""
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _resolve_str(
    *,
    env_key: str,
    env_file: dict[str, str],
    config_value: object,
    default: str,
) -> str:
    if env_key in os.environ and os.environ[env_key].strip():
        return os.environ[env_key].strip()
    if isinstance(config_value, str) and config_value.strip():
        return config_value.strip()
    if env_key in env_file and env_file[env_key].strip():
        return env_file[env_key].strip()
    return default


def _resolve_bool(
    *,
    env_key: str,
    env_file: dict[str, str],
    config_value: object,
    default: bool,
) -> bool:
    if env_key in os.environ:
        return os.environ[env_key].strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(config_value, bool):
        return config_value
    if env_key in env_file:
        return env_file[env_key].strip().lower() in {"1", "true", "yes", "on"}
    return default


def _resolve_storage_path(value: str, project_root: Path, default_path: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve() if str(candidate).strip() else default_path


def load_settings(start: Path | None = None) -> Settings:
    project_root = resolve_project_root(start)
    env_path = project_root / ".env"
    env_file = _parse_env_file(env_path)
    for key, value in env_file.items():
        os.environ.setdefault(key, value)

    config_path = resolve_config_path(project_root)
    ensure_app_dir(project_root)

    if not config_path.exists():
        config_path = init_default_config(start=project_root)

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    storage = raw.get("storage", {})
    llm = raw.get("llm", {})
    retrieval = raw.get("retrieval", {})

    app_dir = resolve_app_dir(project_root)
    default_db_path = resolve_default_db_path(project_root)
    default_lancedb_path = resolve_default_lancedb_path(project_root)
    default_cache_dir = resolve_default_cache_dir(project_root)
    default_logs_dir = resolve_default_logs_dir(project_root)

    storage_db = _resolve_str(
        env_key="PKA_DB_PATH",
        env_file=env_file,
        config_value=storage.get("db_path"),
        default=str(default_db_path),
    )
    db_path = _resolve_storage_path(storage_db, project_root, default_db_path)

    storage_vector = _resolve_str(
        env_key="PKA_VECTORSTORE_PATH",
        env_file=env_file,
        config_value=storage.get("vectorstore_path", storage.get("lancedb_path")),
        default=str(default_lancedb_path),
    )
    lancedb_path = _resolve_storage_path(
        storage_vector, project_root, default_lancedb_path
    )

    cache_dir = _resolve_storage_path(
        _resolve_str(
            env_key="PKA_CACHE_DIR",
            env_file=env_file,
            config_value=storage.get("cache_dir"),
            default=str(default_cache_dir),
        ),
        project_root,
        default_cache_dir,
    )

    logs_dir = _resolve_storage_path(
        _resolve_str(
            env_key="PKA_LOGS_DIR",
            env_file=env_file,
            config_value=storage.get("logs_dir"),
            default=str(default_logs_dir),
        ),
        project_root,
        default_logs_dir,
    )

    llm_provider = _resolve_str(
        env_key="PKA_LLM_PROVIDER",
        env_file=env_file,
        config_value=llm.get("provider", "none"),
        default="none",
    ).lower().strip()

    retrieval_mode = _resolve_str(
        env_key="PKA_RETRIEVAL_MODE",
        env_file=env_file,
        config_value=retrieval.get("mode", "hybrid"),
        default="hybrid",
    ).lower().strip()

    embedding_model = _resolve_str(
        env_key="PKA_EMBEDDING_MODEL",
        env_file=env_file,
        config_value=retrieval.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    return Settings(
        project_root=project_root,
        app_dir=app_dir,
        config_path=config_path,
        db_path=db_path,
        vectorstore_path=lancedb_path,
        cache_dir=cache_dir,
        logs_dir=logs_dir,
        llm_provider=llm_provider,
        ollama_base_url=_resolve_str(
            env_key="OLLAMA_BASE_URL",
            env_file=env_file,
            config_value=llm.get("ollama_base_url", "http://localhost:11434"),
            default="http://localhost:11434",
        ),
        ollama_model=_resolve_str(
            env_key="OLLAMA_MODEL",
            env_file=env_file,
            config_value=llm.get("ollama_model", "llama3.1:8b-instruct-q4_K_M"),
            default="llama3.1:8b-instruct-q4_K_M",
        ),
        openai_model=_resolve_str(
            env_key="OPENAI_MODEL",
            env_file=env_file,
            config_value=llm.get("openai_model", "gpt-4o-mini"),
            default="gpt-4o-mini",
        ),
        anthropic_model=_resolve_str(
            env_key="ANTHROPIC_MODEL",
            env_file=env_file,
            config_value=llm.get("anthropic_model", "claude-3-5-sonnet-latest"),
            default="claude-3-5-sonnet-latest",
        ),
        retrieval_mode=retrieval_mode,
        embedding_model=embedding_model,
        enable_reranker=_resolve_bool(
            env_key="PKA_ENABLE_RERANKER",
            env_file=env_file,
            config_value=retrieval.get("enable_reranker", False),
            default=False,
        ),
        reranker_model=_resolve_str(
            env_key="PKA_RERANKER_MODEL",
            env_file=env_file,
            config_value=retrieval.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
    )


def openai_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return key.strip()

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


APP_DIR = Path.home() / ".pka"
CONFIG_PATH = APP_DIR / "config.toml"
DEFAULT_DB_PATH = APP_DIR / "index.sqlite3"


@dataclass(slots=True)
class Settings:
    db_path: Path = DEFAULT_DB_PATH
    llm_provider: str = "none"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"
    openai_model: str = "gpt-4o-mini"
    retrieval_mode: str = "hybrid"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def ensure_app_dir() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)


def init_default_config(force: bool = False) -> Path:
    ensure_app_dir()
    if CONFIG_PATH.exists() and not force:
        return CONFIG_PATH

    config_text = """[storage]
db_path = \"~/.pka/index.sqlite3\"

[llm]
provider = \"none\"  # none | ollama | openai
ollama_base_url = \"http://localhost:11434\"
ollama_model = \"llama3.1:8b-instruct-q4_K_M\"
openai_model = \"gpt-4o-mini\"

[retrieval]
mode = \"hybrid\"  # lexical | dense | hybrid
embedding_model = \"sentence-transformers/all-MiniLM-L6-v2\"
enable_reranker = false
reranker_model = \"cross-encoder/ms-marco-MiniLM-L-6-v2\"
"""
    CONFIG_PATH.write_text(config_text, encoding="utf-8")
    return CONFIG_PATH


def load_settings() -> Settings:
    ensure_app_dir()
    if not CONFIG_PATH.exists():
        init_default_config()

    raw = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    storage = raw.get("storage", {})
    llm = raw.get("llm", {})
    retrieval = raw.get("retrieval", {})

    db_path = Path(storage.get("db_path", str(DEFAULT_DB_PATH))).expanduser()
    llm_provider = str(llm.get("provider", "none")).lower().strip()

    return Settings(
        db_path=db_path,
        llm_provider=llm_provider,
        ollama_base_url=str(llm.get("ollama_base_url", "http://localhost:11434")).strip(),
        ollama_model=str(llm.get("ollama_model", "llama3.1:8b-instruct-q4_K_M")).strip(),
        openai_model=str(llm.get("openai_model", "gpt-4o-mini")).strip(),
        retrieval_mode=str(retrieval.get("mode", "hybrid")).strip().lower(),
        embedding_model=str(retrieval.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")).strip(),
        enable_reranker=bool(retrieval.get("enable_reranker", False)),
        reranker_model=str(retrieval.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")).strip(),
    )


def openai_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return key.strip()

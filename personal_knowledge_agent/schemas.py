from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class Document:
    source_path: Path
    source_type: str
    title: str
    text: str
    fingerprint: str
    updated_at: datetime


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    source_path: str
    source_type: str
    title: str
    fingerprint: str
    section: str
    text: str
    token_estimate: int
    updated_at: str


@dataclass(slots=True)
class WorkflowContext:
    cwd: str
    git_branch: str | None
    recent_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalResult:
    chunk_id: str
    source_path: str
    title: str
    section: str
    text: str
    score: float
    lexical_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .parsers import parse_by_extension
from .schemas import Chunk, Document


SKIP_FOLDERS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
}


def discover_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_FOLDERS for part in path.parts):
            continue
        files.append(path)
    return files


def compute_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_text(text: str, title: str, source_path: str, source_type: str, fingerprint: str, chunk_size: int = 900) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []
    buffer: list[str] = []
    section = "root"
    running_chars = 0
    index = 0
    updated_at = datetime.now(timezone.utc).isoformat()

    def flush() -> None:
        nonlocal buffer, running_chars, index
        if not buffer:
            return
        content = "\n".join(buffer).strip()
        if not content:
            buffer = []
            running_chars = 0
            return
        chunk_id = f"{fingerprint[:12]}-{index}"
        index += 1
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_path=source_path,
                source_type=source_type,
                title=title,
                fingerprint=fingerprint,
                section=section,
                text=content,
                token_estimate=estimate_tokens(content),
                updated_at=updated_at,
            )
        )
        buffer = []
        running_chars = 0

    for line in lines:
        if line.startswith("#"):
            flush()
            section = line.lstrip("#").strip() or section

        buffer.append(line)
        running_chars += len(line) + 1
        if running_chars >= chunk_size:
            flush()

    flush()
    return chunks


def parse_document(path: Path) -> Document | None:
    parsed = parse_by_extension(path)
    if not parsed:
        return None

    text, source_type = parsed
    text = text.strip()
    if not text:
        return None

    stat = path.stat()
    updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    fingerprint = compute_fingerprint(text)
    return Document(
        source_path=path,
        source_type=source_type,
        title=path.name,
        text=text,
        fingerprint=fingerprint,
        updated_at=updated_at,
    )


def ingest_path(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for file_path in discover_files(root):
        document = parse_document(file_path)
        if not document:
            continue
        doc_chunks = chunk_text(
            text=document.text,
            title=document.title,
            source_path=str(document.source_path),
            source_type=document.source_type,
            fingerprint=document.fingerprint,
        )
        chunks.extend(doc_chunks)
    return chunks

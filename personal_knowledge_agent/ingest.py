from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .parsers import parse_document_with_metadata
from .schemas import AssetRecord, Chunk, ChunkMetadata, Document


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


@dataclass(slots=True)
class IngestBundle:
    chunks: list[Chunk]
    assets: list[AssetRecord]


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


def chunk_text(
    text: str,
    title: str,
    source_path: str,
    source_type: str,
    fingerprint: str,
    *,
    connector: str = "local",
    cloud_url: str | None = None,
    external_id: str | None = None,
    version_id: str | None = None,
    branch_hint: str | None = None,
    chunk_size: int = 900,
) -> list[Chunk]:
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
                connector=connector,
                metadata=ChunkMetadata(
                    cloud_url=cloud_url,
                    external_id=external_id,
                    version_id=version_id,
                    branch_hint=branch_hint,
                ),
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


def _chunk_with_hints(
    *,
    title: str,
    source_path: str,
    source_type: str,
    fingerprint: str,
    connector: str,
    cloud_url: str | None,
    external_id: str | None,
    version_id: str | None,
    branch_hint: str | None,
    hint_items: list,
) -> list[Chunk]:
    updated_at = datetime.now(timezone.utc).isoformat()
    chunks: list[Chunk] = []
    for index, hint in enumerate(hint_items):
        text = (hint.text or "").strip()
        if not text:
            continue
        chunk_id = f"{fingerprint[:12]}-{index}"
        section = (hint.section or (f"Page {hint.page_number}" if hint.page_number else "root")).strip() or "root"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_path=source_path,
                source_type=source_type,
                title=title,
                fingerprint=fingerprint,
                section=section,
                text=text,
                token_estimate=estimate_tokens(text),
                updated_at=updated_at,
                connector=connector,
                metadata=ChunkMetadata(
                    page_number=hint.page_number,
                    section_heading=section,
                    bounding_box=None,
                    cloud_url=cloud_url,
                    image_description=hint.image_description,
                    external_id=external_id,
                    version_id=version_id,
                    branch_hint=branch_hint,
                    source_url=cloud_url,
                ),
            )
        )
    return chunks


def parse_document(path: Path) -> Document | None:
    try:
        parsed = parse_document_with_metadata(path)
    except RuntimeError:
        return None
    if not parsed:
        return None

    text = parsed.text
    source_type = parsed.source_type
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


def ingest_path(root: Path, assets_dir: Path | None = None) -> IngestBundle:
    chunks: list[Chunk] = []
    assets: list[AssetRecord] = []
    for file_path in discover_files(root):
        try:
            parsed = parse_document_with_metadata(file_path, assets_dir=assets_dir)
        except RuntimeError:
            continue

        if not parsed:
            continue

        text = parsed.text.strip()
        if not text:
            continue

        stat = file_path.stat()
        updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        fingerprint = compute_fingerprint(text)

        document = Document(
            source_path=file_path,
            source_type=parsed.source_type,
            title=file_path.name,
            text=text,
            fingerprint=fingerprint,
            updated_at=updated_at,
        )

        if parsed.chunk_hints:
            doc_chunks = _chunk_with_hints(
                title=document.title,
                source_path=str(document.source_path),
                source_type=document.source_type,
                fingerprint=document.fingerprint,
                connector=document.connector,
                cloud_url=document.cloud_url,
                external_id=document.external_id,
                version_id=document.version_id,
                branch_hint=None,
                hint_items=parsed.chunk_hints,
            )
        else:
            doc_chunks = chunk_text(
                text=document.text,
                title=document.title,
                source_path=str(document.source_path),
                source_type=document.source_type,
                fingerprint=document.fingerprint,
                connector=document.connector,
                cloud_url=document.cloud_url,
                external_id=document.external_id,
                version_id=document.version_id,
            )

        for asset_index, asset in enumerate(parsed.assets, start=1):
            asset_id = f"{document.fingerprint[:12]}-asset-{asset_index}"
            assets.append(
                AssetRecord(
                    asset_id=asset_id,
                    source_path=str(document.source_path),
                    title=document.title,
                    asset_type=asset.asset_type,
                    page_number=asset.page_number,
                    file_path=asset.file_path,
                    bbox_json=asset.bbox_json,
                    caption_text=asset.caption_text,
                    ocr_text=asset.ocr_text,
                    figure_id=asset.figure_id,
                )
            )

        chunks.extend(doc_chunks)
    return IngestBundle(chunks=chunks, assets=assets)

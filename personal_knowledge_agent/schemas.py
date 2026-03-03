from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class ChunkMetadata(BaseModel):
    page_number: int | None = None
    bounding_box: BoundingBox | None = None
    slide_number: int | None = None
    section_heading: str | None = None
    cloud_url: str | None = None
    image_description: str | None = None
    external_id: str | None = None
    version_id: str | None = None
    branch_hint: str | None = None
    source_url: str | None = None


class Document(BaseModel):
    source_path: Path
    source_type: str
    title: str
    text: str
    fingerprint: str
    updated_at: datetime
    cloud_url: str | None = None
    external_id: str | None = None
    version_id: str | None = None
    connector: str = "local"
    mime: str | None = None
    tags: list[str] = Field(default_factory=list)


class AssetRecord(BaseModel):
    asset_id: str
    source_path: str
    title: str
    asset_type: str
    page_number: int | None = None
    file_path: str | None = None
    bbox_json: str | None = None
    caption_text: str | None = None
    ocr_text: str | None = None
    figure_id: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    source_path: str
    source_type: str
    title: str
    fingerprint: str
    section: str
    text: str
    token_estimate: int
    updated_at: str
    connector: str = "local"
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)


class WorkflowContext(BaseModel):
    cwd: str
    project_root: str
    git_root: str | None = None
    git_branch: str | None
    recent_files: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    chunk_id: str
    source_path: str
    title: str
    section: str
    text: str
    score: float
    lexical_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0
    source_branch: str | None = None
    cloud_url: str | None = None
    workflow_boost: float = 1.0
    page_number: int | None = None
    section_heading: str | None = None
    slide_number: int | None = None
    source_url: str | None = None
    selected_reason: str | None = None

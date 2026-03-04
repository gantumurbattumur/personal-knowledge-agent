from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path


TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
    ".zsh",
}


@dataclass(slots=True)
class ParsedChunkHint:
    text: str
    page_number: int | None = None
    section: str | None = None
    bounding_box_json: str | None = None
    image_description: str | None = None


@dataclass(slots=True)
class ParsedAsset:
    asset_type: str
    page_number: int | None
    file_path: str | None
    caption_text: str | None = None
    ocr_text: str | None = None
    figure_id: str | None = None
    bbox_json: str | None = None


@dataclass(slots=True)
class ParsedDocumentResult:
    text: str
    source_type: str
    chunk_hints: list[ParsedChunkHint] = field(default_factory=list)
    assets: list[ParsedAsset] = field(default_factory=list)


def parse_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _ocr_image_bytes(image_bytes: bytes) -> str | None:
    try:
        import pytesseract
        from PIL import Image
    except ModuleNotFoundError:
        return None

    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image).strip()
        return text or None
    except Exception:
        return None


def parse_pdf(path: Path, assets_dir: Path | None = None) -> ParsedDocumentResult:
    chunk_hints: list[ParsedChunkHint] = []
    assets: list[ParsedAsset] = []

    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(path))
        if hasattr(result, "document") and hasattr(result.document, "export_to_markdown"):
            markdown = result.document.export_to_markdown()
            if markdown and markdown.strip():
                sections = [segment.strip() for segment in markdown.split("\n\n") if segment.strip()]
                for idx, segment in enumerate(sections, start=1):
                    section = "root"
                    first = segment.splitlines()[0].strip() if segment.splitlines() else ""
                    if first.startswith("#"):
                        section = first.lstrip("#").strip() or "root"
                    page_hint = idx if idx <= 500 else None
                    chunk_hints.append(
                        ParsedChunkHint(
                            text=segment,
                            page_number=page_hint,
                            section=section,
                        )
                    )

                full_text = markdown.strip()
                return ParsedDocumentResult(
                    text=full_text,
                    source_type="pdf",
                    chunk_hints=chunk_hints,
                    assets=assets,
                )
    except ModuleNotFoundError:
        pass
    except Exception:
        pass

    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("PDF parsing requires optional dependency: uv pip install -e .[multimodal]") from exc

    resolved_assets = assets_dir.expanduser().resolve() if assets_dir else None
    if resolved_assets:
        resolved_assets.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            chunk_hints.append(
                ParsedChunkHint(
                    text=page_text,
                    page_number=page_index,
                    section=f"Page {page_index}",
                )
            )
            pages.append(f"## Page {page_index}\n{page_text}")

        try:
            extracted_images = list(getattr(page, "images", []) or [])
        except Exception:
            extracted_images = []
        for image_index, image in enumerate(extracted_images, start=1):
            image_bytes = getattr(image, "data", None)
            if not image_bytes:
                continue

            filename = getattr(image, "name", f"page-{page_index}-image-{image_index}.bin")
            digest = hashlib.sha1(f"{path}:{page_index}:{image_index}:{filename}".encode("utf-8")).hexdigest()[:12]
            extension = Path(filename).suffix or ".bin"
            saved_path: Path | None = None
            if resolved_assets:
                saved_path = resolved_assets / f"{path.stem}-p{page_index}-img{image_index}-{digest}{extension}"
                saved_path.write_bytes(image_bytes)

            ocr_text = _ocr_image_bytes(image_bytes)
            caption = f"Image on page {page_index}" if not ocr_text else f"Image on page {page_index}: {ocr_text[:240]}"
            assets.append(
                ParsedAsset(
                    asset_type="image",
                    page_number=page_index,
                    file_path=str(saved_path) if saved_path else None,
                    caption_text=caption,
                    ocr_text=ocr_text,
                    figure_id=f"p{page_index}-img{image_index}",
                )
            )
            chunk_hints.append(
                ParsedChunkHint(
                    text=f"Figure {page_index}.{image_index}: {caption}",
                    page_number=page_index,
                    section=f"Figure p{page_index}-img{image_index}",
                    image_description=caption,
                )
            )

    if not pages and not chunk_hints:
        return ParsedDocumentResult(text="", source_type="pdf", chunk_hints=[], assets=[])

    full_text = "\n\n".join(pages).strip()
    return ParsedDocumentResult(
        text=full_text,
        source_type="pdf",
        chunk_hints=chunk_hints,
        assets=assets,
    )


def parse_epub(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        from ebooklib import ITEM_DOCUMENT, epub
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "EPUB parsing requires optional dependencies: uv pip install -e .[epub]"
        ) from exc

    book = epub.read_epub(str(path))
    text_chunks: list[str] = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if text:
            text_chunks.append(text)
    return "\n\n".join(text_chunks).strip()


def parse_by_extension(path: Path) -> tuple[str, str] | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return parse_text_file(path), "text"
    if suffix == ".pdf":
        return parse_pdf(path).text, "pdf"
    if suffix == ".epub":
        return parse_epub(path), "epub"
    return None


def parse_document_with_metadata(path: Path, assets_dir: Path | None = None) -> ParsedDocumentResult | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        text = parse_text_file(path)
        return ParsedDocumentResult(text=text, source_type="text")
    if suffix == ".pdf":
        return parse_pdf(path, assets_dir=assets_dir)
    if suffix == ".epub":
        text = parse_epub(path)
        return ParsedDocumentResult(text=text, source_type="epub")
    return None

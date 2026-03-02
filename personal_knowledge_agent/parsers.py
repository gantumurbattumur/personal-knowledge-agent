from __future__ import annotations

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


def parse_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError("PDF parsing requires optional dependency: pip install personal-knowledge-agent[pdf]") from exc

    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks).strip()


def parse_epub(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        from ebooklib import ITEM_DOCUMENT, epub
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "EPUB parsing requires optional dependencies: pip install personal-knowledge-agent[epub]"
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
        return parse_pdf(path), "pdf"
    if suffix == ".epub":
        return parse_epub(path), "epub"
    return None

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib import request
import json
from pathlib import Path

from .index_store import IndexStore


@dataclass(slots=True)
class NotionPage:
    page_id: str
    title: str
    markdown: str
    version_id: str | None
    cloud_url: str | None


@dataclass(slots=True)
class NotionSyncResult:
    ok: bool
    destination: Path
    downloaded: int
    unchanged: int
    message: str


def _request_json(url: str, token: str, method: str = "GET", payload: dict | None = None, retries: int = 4) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url=url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Notion-Version", "2022-06-28")
    req.add_header("Content-Type", "application/json")

    backoff = 1.0
    for _ in range(retries):
        try:
            with request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            status = getattr(exc, "code", None)
            if status == 429:
                retry_after = getattr(exc, "headers", {}).get("Retry-After") if hasattr(exc, "headers") else None
                sleep_for = float(retry_after) if retry_after else backoff
                time.sleep(sleep_for)
                backoff = min(backoff * 2, 10)
                continue
            raise
    raise RuntimeError("Notion request failed after retries")


def rich_text_to_markdown(parts: list[dict]) -> str:
    out: list[str] = []
    for part in parts:
        text = part.get("plain_text", "")
        ann = part.get("annotations", {})
        if ann.get("code"):
            text = f"`{text}`"
        if ann.get("bold"):
            text = f"**{text}**"
        if ann.get("italic"):
            text = f"*{text}*"
        out.append(text)
    return "".join(out)


def block_to_markdown(block: dict) -> str:
    block_type = block.get("type", "")
    payload = block.get(block_type, {})
    rich_text = payload.get("rich_text", [])
    text = rich_text_to_markdown(rich_text)

    if block_type == "heading_1":
        return f"# {text}" if text else ""
    if block_type == "heading_2":
        return f"## {text}" if text else ""
    if block_type == "heading_3":
        return f"### {text}" if text else ""
    if block_type == "bulleted_list_item":
        return f"- {text}" if text else ""
    if block_type == "numbered_list_item":
        return f"1. {text}" if text else ""
    if block_type == "to_do":
        checked = payload.get("checked", False)
        box = "x" if checked else " "
        return f"- [{box}] {text}" if text else ""
    if block_type == "quote":
        return f"> {text}" if text else ""
    if block_type == "code":
        lang = payload.get("language", "text")
        return f"```{lang}\n{text}\n```" if text else ""
    if block_type == "paragraph":
        return text
    return text


def blocks_to_markdown(blocks: list[dict]) -> str:
    lines = [block_to_markdown(block) for block in blocks]
    return "\n".join(line for line in lines if line).strip()


def _extract_title(page_payload: dict) -> str:
    props = page_payload.get("properties", {})
    for value in props.values():
        if value.get("type") == "title":
            text = rich_text_to_markdown(value.get("title", []))
            if text:
                return text
    return page_payload.get("id", "untitled")


def _fetch_all_blocks(page_id: str, token: str) -> list[dict]:
    blocks: list[dict] = []
    cursor: str | None = None
    while True:
        url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"
        if cursor:
            url += f"&start_cursor={cursor}"
        payload = _request_json(url, token=token, method="GET")
        blocks.extend(payload.get("results", []))
        if not payload.get("has_more"):
            break
        cursor = payload.get("next_cursor")
    return blocks


def fetch_page(page_id: str, token: str) -> NotionPage:
    page = _request_json(f"https://api.notion.com/v1/pages/{page_id}", token=token, method="GET")
    title = _extract_title(page)
    version_id = page.get("last_edited_time")
    cloud_url = page.get("url")
    blocks = _fetch_all_blocks(page_id, token)
    markdown = blocks_to_markdown(blocks)
    if not markdown:
        markdown = title
    return NotionPage(
        page_id=page_id,
        title=title,
        markdown=markdown,
        version_id=version_id,
        cloud_url=cloud_url,
    )


def sync_notion_incremental(
    *,
    root_page_id: str,
    destination: Path,
    token_payload: dict,
    store: IndexStore,
) -> NotionSyncResult:
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    token = token_payload.get("access_token") or token_payload.get("token")
    if not token:
        return NotionSyncResult(
            ok=False,
            destination=destination,
            downloaded=0,
            unchanged=0,
            message="Notion token payload is missing access token",
        )

    state = store.get_sync_state("notion")

    try:
        page = fetch_page(root_page_id, token)
    except Exception as exc:
        return NotionSyncResult(
            ok=False,
            destination=destination,
            downloaded=0,
            unchanged=0,
            message=f"Notion sync failed: {exc}",
        )

    known = state.get(page.page_id)
    known_version = known.get("version_id") if known else None
    if known and known_version == page.version_id:
        return NotionSyncResult(
            ok=True,
            destination=destination,
            downloaded=0,
            unchanged=1,
            message="Notion page unchanged",
        )

    safe_name = "".join(char if char.isalnum() or char in {"-", "_", " "} else "_" for char in page.title).strip()
    file_name = (safe_name or page.page_id) + ".md"
    local_path = destination / file_name
    local_path.write_text(page.markdown, encoding="utf-8")

    store.upsert_sync_state(
        connector="notion",
        external_id=page.page_id,
        version_id=page.version_id,
        source_path=str(local_path),
        cloud_url=page.cloud_url,
    )

    return NotionSyncResult(
        ok=True,
        destination=destination,
        downloaded=1,
        unchanged=0,
        message="Notion sync completed",
    )

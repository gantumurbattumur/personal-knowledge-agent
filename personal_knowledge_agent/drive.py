from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .index_store import IndexStore


@dataclass(slots=True)
class DriveSyncResult:
    ok: bool
    destination: Path
    file_count: int
    message: str


@dataclass(slots=True)
class DriveRemoteFile:
    file_id: str
    name: str
    version_id: str | None
    cloud_url: str | None = None


@dataclass(slots=True)
class DriveIncrementalPlan:
    to_download: list[DriveRemoteFile]
    unchanged: list[DriveRemoteFile]


@dataclass(slots=True)
class IncrementalSyncResult:
    ok: bool
    destination: Path
    downloaded: int
    unchanged: int
    message: str


def plan_incremental_sync(store: IndexStore, remote_files: list[DriveRemoteFile]) -> DriveIncrementalPlan:
    state = store.get_sync_state("google-drive")
    to_download: list[DriveRemoteFile] = []
    unchanged: list[DriveRemoteFile] = []

    for remote in remote_files:
        known = state.get(remote.file_id)
        known_version = known.get("version_id") if known else None
        if not known or known_version != remote.version_id:
            to_download.append(remote)
        else:
            unchanged.append(remote)
    return DriveIncrementalPlan(to_download=to_download, unchanged=unchanged)


def record_sync_state(store: IndexStore, remote_file: DriveRemoteFile, local_path: Path) -> None:
    store.upsert_sync_state(
        connector="google-drive",
        external_id=remote_file.file_id,
        version_id=remote_file.version_id,
        source_path=str(local_path),
        cloud_url=remote_file.cloud_url,
    )


def sync_google_drive_incremental(
    *,
    folder_id: str,
    destination: Path,
    token_payload: dict,
    store: IndexStore,
) -> IncrementalSyncResult:
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ModuleNotFoundError:
        return IncrementalSyncResult(
            ok=False,
            destination=destination,
            downloaded=0,
            unchanged=0,
            message="Google connector dependencies are missing. Install with: uv pip install -e .[connectors]",
        )

    try:
        credentials = Credentials(
            token=token_payload.get("token"),
            refresh_token=token_payload.get("refresh_token"),
            token_uri=token_payload.get("token_uri"),
            client_id=token_payload.get("client_id"),
            client_secret=token_payload.get("client_secret"),
            scopes=token_payload.get("scopes"),
        )
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    except Exception as exc:
        return IncrementalSyncResult(
            ok=False,
            destination=destination,
            downloaded=0,
            unchanged=0,
            message=f"Failed to initialize Google Drive client: {exc}",
        )

    query = f"'{folder_id}' in parents and trashed = false"
    remote_files: list[DriveRemoteFile] = []
    page_token: str | None = None

    while True:
        response = (
            service.files()
            .list(
                q=query,
                pageSize=100,
                fields="nextPageToken, files(id,name,version,webViewLink,mimeType)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for item in response.get("files", []):
            remote_files.append(
                DriveRemoteFile(
                    file_id=str(item.get("id")),
                    name=str(item.get("name")),
                    version_id=str(item.get("version")) if item.get("version") is not None else None,
                    cloud_url=item.get("webViewLink"),
                )
            )
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    plan = plan_incremental_sync(store, remote_files)

    downloaded = 0
    for remote in plan.to_download:
        local_path = destination / remote.name
        request_handle = service.files().get_media(fileId=remote.file_id)
        with local_path.open("wb") as stream:
            downloader = MediaIoBaseDownload(stream, request_handle)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        record_sync_state(store, remote, local_path)
        downloaded += 1

    return IncrementalSyncResult(
        ok=True,
        destination=destination,
        downloaded=downloaded,
        unchanged=len(plan.unchanged),
        message="Incremental sync completed",
    )


def sync_public_folder(url: str, destination: Path, quiet: bool = True) -> DriveSyncResult:
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ModuleNotFoundError:
        return DriveSyncResult(
            ok=False,
            destination=destination,
            file_count=0,
            message="gdown is not installed. Install with: uv pip install -e .[gdrive]",
        )

    try:
        downloaded_files = gdown.download_folder(
            url=url,
            output=str(destination),
            quiet=quiet,
            resume=True,
            use_cookies=False,
            remaining_ok=True,
        )
    except Exception as exc:
        return DriveSyncResult(
            ok=False,
            destination=destination,
            file_count=0,
            message=f"Google Drive sync failed: {exc}",
        )

    count = len(downloaded_files or [])
    return DriveSyncResult(
        ok=True,
        destination=destination,
        file_count=count,
        message="Sync completed",
    )

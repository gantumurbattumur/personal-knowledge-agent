from __future__ import annotations

import json
import importlib
from dataclasses import dataclass
from urllib import parse, request


KEYRING_APP = "personal-knowledge-agent"


@dataclass(slots=True)
class AuthResult:
    ok: bool
    service: str
    message: str


def _load_keyring():
    try:
        keyring = importlib.import_module("keyring")
    except ModuleNotFoundError as exc:
        raise RuntimeError("keyring is not installed. Install with: uv pip install -e .[auth]") from exc
    return keyring


def save_token(service: str, payload: dict) -> None:
    keyring = _load_keyring()
    keyring.set_password(KEYRING_APP, service, json.dumps(payload))


def load_token(service: str) -> dict | None:
    keyring = _load_keyring()
    raw = keyring.get_password(KEYRING_APP, service)
    if not raw:
        return None
    return json.loads(raw)


def auth_google_drive(client_secret_file: str) -> AuthResult:
    try:
        flow_module = importlib.import_module("google_auth_oauthlib.flow")
        InstalledAppFlow = flow_module.InstalledAppFlow
    except ModuleNotFoundError:
        return AuthResult(
            ok=False,
            service="google-drive",
            message="Google OAuth dependencies are missing. Install with: uv pip install -e .[connectors]",
        )

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    try:
        flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, scopes=scopes)
        creds = flow.run_local_server(port=0)
        save_token(
            "google-drive",
            {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": list(creds.scopes or scopes),
            },
        )
    except Exception as exc:
        return AuthResult(ok=False, service="google-drive", message=f"OAuth failed: {exc}")

    return AuthResult(ok=True, service="google-drive", message="Google Drive auth completed and token saved to keyring")


def auth_notion(client_id: str, client_secret: str, redirect_uri: str, code: str) -> AuthResult:
    token_url = "https://api.notion.com/v1/oauth/token"
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(token_url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Basic " + _basic_auth(client_id, client_secret))

    try:
        with request.urlopen(req, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return AuthResult(ok=False, service="notion", message=f"Notion OAuth failed: {exc}")

    save_token("notion", body)
    return AuthResult(ok=True, service="notion", message="Notion auth completed and token saved to keyring")


def notion_authorize_url(client_id: str, redirect_uri: str, state: str = "pka") -> str:
    params = parse.urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "owner": "user",
            "redirect_uri": redirect_uri,
            "state": state,
        }
    )
    return f"https://api.notion.com/v1/oauth/authorize?{params}"


def _basic_auth(client_id: str, client_secret: str) -> str:
    import base64

    raw = f"{client_id}:{client_secret}".encode("utf-8")
    return base64.b64encode(raw).decode("ascii")

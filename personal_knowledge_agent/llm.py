from __future__ import annotations

import json
from urllib import request

from .config import Settings, openai_api_key


def _post_json(url: str, payload: dict, headers: dict[str, str] | None = None, timeout: int = 45) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    req = request.Request(url=url, data=body, headers=req_headers, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _prompt(question: str, evidence: str) -> str:
    return (
        "Answer the question using only the provided evidence. "
        "If evidence is insufficient, say so briefly.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence}\n"
    )


def generate_answer(settings: Settings, question: str, evidence: str) -> str:
    provider = settings.llm_provider
    if provider == "none":
        return evidence
    if provider == "ollama":
        return _answer_ollama(settings, question, evidence)
    if provider == "openai":
        return _answer_openai(settings, question, evidence)
    return evidence


def _answer_ollama(settings: Settings, question: str, evidence: str) -> str:
    payload = {
        "model": settings.ollama_model,
        "prompt": _prompt(question, evidence),
        "stream": False,
    }
    url = settings.ollama_base_url.rstrip("/") + "/api/generate"
    try:
        data = _post_json(url, payload)
        text = str(data.get("response", "")).strip()
        return text or evidence
    except Exception:
        return evidence


def _answer_openai(settings: Settings, question: str, evidence: str) -> str:
    api_key = openai_api_key()
    if not api_key:
        return evidence

    payload = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": "You are a grounded assistant that only uses user-provided evidence."},
            {"role": "user", "content": _prompt(question, evidence)},
        ],
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        data = _post_json("https://api.openai.com/v1/chat/completions", payload, headers=headers)
        content = data["choices"][0]["message"]["content"]
        return str(content).strip() or evidence
    except Exception:
        return evidence

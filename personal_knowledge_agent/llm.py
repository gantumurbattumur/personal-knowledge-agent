from __future__ import annotations

import json
import re
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


def _query_cleanup(text: str) -> str:
    lowered = text.strip()
    if not lowered:
        return ""
    normalized = re.sub(r"\s+", " ", lowered)
    normalized = normalized.replace("\n", " ").strip()
    return normalized


def _heuristic_query_rewrite(question: str) -> str:
    cleaned = _query_cleanup(question)
    if not cleaned:
        return question.strip()

    lowered = cleaned.lower()
    noise_phrases = [
        "tell me",
        "please",
        "can you",
        "could you",
        "i want",
        "give me",
    ]
    for phrase in noise_phrases:
        lowered = lowered.replace(phrase, " ")

    lowered = re.sub(r"[^a-z0-9\s\-]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()

    if not lowered:
        return cleaned

    if "genai" in lowered:
        lowered = lowered.replace("genai", "generative ai")

    return lowered


def detect_query_noise(query: str) -> dict[str, bool]:
    """Detect grammatical errors, typos, and noise in query."""
    noise_indicators = {
        "has_extra_punctuation": bool(re.search(r"[?!]{2,}", query)),
        "has_mixed_case_error": bool(re.search(r"\b[a-z]+[A-Z][a-z]*\b", query)),  # camelCase in text
        "has_excessive_spaces": bool(re.search(r"  {2,}", query)),
        "likely_typo": len(query) > 0 and any(
            word in query.lower() 
            for word in ["rnns", "rnn's", "transformerss", "gpt4", "claud", "llama2", "mistrl"]
        ),
        "is_very_short": len(query.split()) < 2,
        "is_very_long": len(query.split()) > 50,
    }
    return noise_indicators


def normalize_query(query: str) -> str:
    """Normalize and fix common query issues."""
    text = _query_cleanup(query)
    if not text:
        return query.strip()

    # Lemmatize/normalize common technical terms
    term_map = {
        r"\brnns\b": "RNN",
        r"\brnn's\b": "RNN",
        r"\btransformers\b": "transformers",
        r"\btransformer's\b": "transformer",
        r"\bgpt4\b": "GPT-4",
        r"\bgpt-3\b": "GPT-3",
        r"\bgpt-4\b": "GPT-4",
        r"\bclaud\b": "Claude",
        r"\bclaude3\b": "Claude 3",
        r"\bllama2\b": "Llama 2",
        r"\bmistrl\b": "Mistral",
    }

    lowered = text.lower()
    for pattern, replacement in term_map.items():
        lowered = re.sub(pattern, replacement, lowered, flags=re.IGNORECASE)

    # Fix spacing around punctuation
    normalized = re.sub(r"\s+([?!,.])", r"\1", lowered)
    normalized = re.sub(r"([?!,.])\s*([a-z])", r"\1 \2", normalized)

    # Remove excess punctuation
    normalized = re.sub(r"[?!]{2,}", "?", normalized)
    normalized = re.sub(r"\.{2,}", ".", normalized)

    # Collapse extra whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized if normalized else text


def optimize_query_for_retrieval(settings: Settings, question: str, context_hint: str | None = None) -> str:
    # First, normalize the query if enabled
    if settings.enable_query_normalization:
        question = normalize_query(question)
    
    provider = settings.llm_provider
    fallback = _heuristic_query_rewrite(question)

    if provider in {"none", ""}:
        return fallback
    if provider in {"local", "ollama"}:
        optimized = _optimize_query_ollama(settings, question, context_hint)
        return optimized or fallback
    if provider == "openai":
        optimized = _optimize_query_openai(settings, question, context_hint)
        return optimized or fallback
    return fallback


def _query_rewrite_prompt(question: str, context_hint: str | None = None) -> str:
    hint = context_hint.strip() if context_hint else ""
    prompt = (
        "Rewrite the user question into a concise retrieval query for document search. "
        "Preserve intent and key constraints. Remove conversational filler. "
        "Output only one single-line query string and no explanation.\n\n"
        f"Question: {question.strip()}"
    )
    if hint:
        prompt += f"\nContext hint: {hint}"
    return prompt


def _optimize_query_ollama(settings: Settings, question: str, context_hint: str | None = None) -> str | None:
    payload = {
        "model": settings.ollama_model,
        "prompt": _query_rewrite_prompt(question, context_hint),
        "stream": False,
        "options": {"temperature": 0.0},
    }
    url = settings.ollama_base_url.rstrip("/") + "/api/generate"
    try:
        data = _post_json(url, payload)
        text = _query_cleanup(str(data.get("response", "")))
        return text or None
    except Exception:
        return None


def _optimize_query_openai(settings: Settings, question: str, context_hint: str | None = None) -> str | None:
    api_key = openai_api_key()
    if not api_key:
        return None

    payload = {
        "model": settings.openai_model,
        "messages": [
            {
                "role": "system",
                "content": "You optimize retrieval queries for RAG. Return only the rewritten query string.",
            },
            {
                "role": "user",
                "content": _query_rewrite_prompt(question, context_hint),
            },
        ],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        data = _post_json("https://api.openai.com/v1/chat/completions", payload, headers=headers)
        content = str(data["choices"][0]["message"]["content"])
        text = _query_cleanup(content)
        return text or None
    except Exception:
        return None


def generate_answer(settings: Settings, question: str, evidence: str) -> str:
    provider = settings.llm_provider
    if provider == "none":
        return evidence
    if provider in {"ollama", "local"}:
        return _answer_ollama(settings, question, evidence)
    if provider == "openai":
        return _answer_openai(settings, question, evidence)
    if provider == "anthropic":
        return evidence
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

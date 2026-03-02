from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EmbeddingStatus:
    enabled: bool
    reason: str


class EmbeddingEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._enabled = False
        self._reason = "uninitialized"
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError:
            self._enabled = False
            self._reason = "sentence-transformers is not installed"
            return

        try:
            self._model = SentenceTransformer(self.model_name)
            self._enabled = True
            self._reason = "ok"
        except Exception as exc:
            self._enabled = False
            self._reason = f"failed to load model: {exc}"

    @property
    def status(self) -> EmbeddingStatus:
        return EmbeddingStatus(enabled=self._enabled, reason=self._reason)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not self._enabled or self._model is None:
            return []

        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def encode_query(self, text: str) -> list[float] | None:
        vectors = self.encode([text])
        if not vectors:
            return None
        return vectors[0]


class RerankerEngine:
    def __init__(self, model_name: str, enabled: bool):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._reason = "disabled"
        if enabled:
            self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError:
            self.enabled = False
            self._reason = "sentence-transformers is not installed"
            return

        try:
            self._model = CrossEncoder(self.model_name)
            self._reason = "ok"
        except Exception as exc:
            self.enabled = False
            self._reason = f"failed to load model: {exc}"

    @property
    def status(self) -> EmbeddingStatus:
        return EmbeddingStatus(enabled=self.enabled, reason=self._reason)

    def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        if not self.enabled or self._model is None:
            return None
        pairs = [(query, passage) for passage in passages]
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]

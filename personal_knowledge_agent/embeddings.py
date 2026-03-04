from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from urllib import request


@dataclass(slots=True)
class EmbeddingStatus:
    enabled: bool
    reason: str


@contextlib.contextmanager
def _suppress_model_load_output():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


class EmbeddingEngine:
    """Base embedding engine supporting local and API-based models."""
    
    def __init__(self, model_name: str, provider: str = "local"):
        self.model_name = model_name
        self.provider = provider
        self._model = None
        self._enabled = False
        self._reason = "uninitialized"
        self._dims = 384  # default for MiniLM
        self._load()

    def _load(self) -> None:
        """Load embedding model based on provider."""
        if self.provider == "local":
            self._load_local()
        elif self.provider == "openai":
            self._load_openai()
        elif self.provider == "jina":
            self._load_jina()
        else:
            self._load_local()

    def _load_local(self) -> None:
        """Load local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError:
            self._enabled = False
            self._reason = "sentence-transformers is not installed"
            return

        try:
            with _suppress_model_load_output():
                self._model = SentenceTransformer(self.model_name)
                # Get embedding dimension
                dummy = self._model.encode(["test"])
                self._dims = len(dummy[0]) if len(dummy) > 0 else 384
            self._enabled = True
            self._reason = "ok"
        except Exception as exc:
            self._enabled = False
            self._reason = f"failed to load model: {exc}"

    def _load_openai(self) -> None:
        """Check OpenAI API availability."""
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self._enabled = False
            self._reason = "OPENAI_API_KEY not set"
            return
        
        self._enabled = True
        self._reason = "ok"
        self._dims = 1536  # OpenAI's text-embedding-3-small
        if "large" in self.model_name.lower():
            self._dims = 3072
        self._model = api_key  # store key for later use

    def _load_jina(self) -> None:
        """Check Jina API availability."""
        import os
        
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            self._enabled = False
            self._reason = "JINA_API_KEY not set"
            return
        
        self._enabled = True
        self._reason = "ok"
        self._dims = 768  # Jina's default
        self._model = api_key  # store key for later use

    @property
    def status(self) -> EmbeddingStatus:
        return EmbeddingStatus(enabled=self._enabled, reason=self._reason)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to embeddings."""
        if not self._enabled or self._model is None:
            return []

        if self.provider == "local":
            return self._encode_local(texts)
        elif self.provider == "openai":
            return self._encode_openai(texts)
        elif self.provider == "jina":
            return self._encode_jina(texts)
        return []

    def _encode_local(self, texts: list[str]) -> list[list[float]]:
        """Encode using local sentence-transformers."""
        if self._model is None:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def _encode_openai(self, texts: list[str]) -> list[list[float]]:
        """Encode using OpenAI API."""
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return []
        
        model = "text-embedding-3-small"
        if "large" in self.model_name.lower():
            model = "text-embedding-3-large"

        try:
            payload = {
                "model": model,
                "input": texts,
                "encoding_format": "float",
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(
                "https://api.openai.com/v1/embeddings",
                data=body,
                headers=headers,
                method="POST",
            )
            with request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            
            # Extract embeddings in order
            embeddings_data = result.get("data", [])
            embeddings_data.sort(key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in embeddings_data]
        except Exception:
            return []

    def _encode_jina(self, texts: list[str]) -> list[list[float]]:
        """Encode using Jina API."""
        import os
        
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            return []

        try:
            payload = {
                "model": self.model_name,
                "input": texts,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(
                "https://api.jina.ai/v1/embeddings",
                data=body,
                headers=headers,
                method="POST",
            )
            with request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            
            embeddings_data = result.get("data", [])
            embeddings_data.sort(key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in embeddings_data]
        except Exception:
            return []

    def encode_query(self, text: str) -> list[float] | None:
        """Encode single query text."""
        vectors = self.encode([text])
        if not vectors:
            return None
        return vectors[0]

    @property
    def embedding_dims(self) -> int:
        """Get embedding dimensionality."""
        return self._dims


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
            with _suppress_model_load_output():
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

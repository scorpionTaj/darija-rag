"""Minimal client for any OpenAI-compatible chat endpoint: LM Studio, Bionic, Ollama, vLLM,
OpenRouter or your own gateway.

Configured from the environment:
    DARIJA_RAG_LLM_URL    base URL ending in /v1 (default: LM Studio, http://localhost:1234/v1)
    DARIJA_RAG_LLM_MODEL  model id; empty means the first model the server lists
    DARIJA_RAG_LLM_KEY    bearer token, if the endpoint needs one
"""

import os
from dataclasses import dataclass, field

import httpx

DEFAULT_URL = "http://localhost:1234/v1"


class LLMError(RuntimeError):
    pass


@dataclass
class LLMClient:
    base_url: str = DEFAULT_URL
    model: str = ""
    api_key: str = ""
    timeout: float = 120.0
    _http: httpx.Client | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls(
            base_url=os.environ.get("DARIJA_RAG_LLM_URL", DEFAULT_URL).rstrip("/"),
            model=os.environ.get("DARIJA_RAG_LLM_MODEL", ""),
            api_key=os.environ.get("DARIJA_RAG_LLM_KEY", ""),
        )

    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout)
        return self._http

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"LLM endpoint {self.base_url} is unreachable ({exc.__class__.__name__}). "
                "Start the LM Studio server (Developer tab) or set DARIJA_RAG_LLM_URL."
            ) from exc
        if response.status_code >= 400:
            raise LLMError(f"LLM endpoint returned {response.status_code}: {response.text[:300]}")
        return response.json()

    def resolve_model(self) -> str:
        if not self.model:
            self.model = self._loaded_lm_studio_model() or self._first_listed_model()
        return self.model

    def _loaded_lm_studio_model(self) -> str | None:
        """LM Studio's /v1/models lists every downloaded model; its native /api/v0/models says
        which are loaded. Prefer a loaded chat model with a known architecture (stubs like TTS
        models report none)."""
        root = self.base_url.removesuffix("/v1")
        try:
            data = self.http.get(f"{root}/api/v0/models", timeout=5).json().get("data", [])
        except (httpx.HTTPError, ValueError):
            return None  # not LM Studio
        loaded = [m for m in data if m.get("state") == "loaded" and m.get("type") in ("llm", "vlm")]
        loaded.sort(key=lambda m: m.get("arch") is None)
        return loaded[0]["id"] if loaded else None

    def _first_listed_model(self) -> str:
        models = self._request("GET", "/models").get("data", [])
        chat_models = [m["id"] for m in models if "embed" not in m["id"].lower()]
        if not chat_models:
            raise LLMError(
                f"no chat model at {self.base_url}; load one or set DARIJA_RAG_LLM_MODEL"
            )
        return chat_models[0]

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 600) -> str:
        body = {
            "model": self.resolve_model(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._request("POST", "/chat/completions", json=body)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from exc

from __future__ import annotations

import json
import urllib.request
from types import SimpleNamespace
from typing import Any


class _ChatCompletions:
    def __init__(self, client: "OpenAI"):
        self.client = client
    def create(self, **kwargs):
        payload = self.client._post("/chat/completions", kwargs)
        choices = []
        for choice in payload.get("choices", []):
            msg = choice.get("message", {})
            choices.append(SimpleNamespace(message=SimpleNamespace(content=msg.get("content", ""), reasoning_content=msg.get("reasoning_content", ""))))
        return SimpleNamespace(choices=choices)


class _Chat:
    def __init__(self, client: "OpenAI"):
        self.completions = _ChatCompletions(client)


class _Embeddings:
    def __init__(self, client: "OpenAI"):
        self.client = client
    def create(self, **kwargs):
        payload = self.client._post("/embeddings", kwargs)
        items = [SimpleNamespace(embedding=item.get("embedding", [])) for item in payload.get("data", [])]
        return SimpleNamespace(data=items)


class OpenAI:
    """Minimal OpenAI-compatible HTTP client used only when openai package is absent."""
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float | None = 600.0, **_: Any):
        self.api_key = api_key or "EMPTY"
        self.base_url = (base_url or "http://127.0.0.1:8000/v1").rstrip("/")
        self.timeout = timeout
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)
    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class AsyncOpenAI(OpenAI):
    async def aclose(self):
        return None

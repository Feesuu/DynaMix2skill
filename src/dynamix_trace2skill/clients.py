from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .tokenization import TokenizerUnavailable, truncate_with_tokenizer


@dataclass
class GenerationConfig:
    base_url: str = "mock://deterministic"
    model: str = "Qwen3.5-9B"
    api_key: str = "EMPTY"
    temperature: float = 0.6
    timeout_seconds: float = 600.0
    max_concurrency: int = 4
    thinking_mode: bool | None = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    debug_dir: str | None = None
    retry_wait_seconds: tuple[float, ...] = (2.0, 5.0, 15.0)


@dataclass
class EmbeddingConfig:
    base_url: str = "mock://deterministic"
    model: str = "Qwen3-Embedding-8B"
    api_key: str = "EMPTY"
    # Hard maximum input length for the embedding model.  For Qwen3-Embedding-8B
    # the project contract is 32k tokens.  We do not chunk+pool in this v1
    # Trace2Skill-reuse path; if a trace exceeds the limit, it is truncated to
    # this configured maximum and the truncation is reported as a run artifact.
    max_model_len: int = 32000
    max_input_tokens: int | None = None
    truncate_long_texts: bool = True
    tokenizer_model: str | None = None
    tokenizer_required: bool = True
    truncation_strategy: str = "head"
    batch_size: int = 8
    max_concurrency: int = 4
    cache_path: str | None = None
    deterministic_dim: int = 384

    @property
    def effective_max_input_tokens(self) -> int:
        return int(self.max_input_tokens or self.max_model_len)


class GenerationClient:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self._sem = asyncio.Semaphore(max(1, int(config.max_concurrency)))
        self._counter = 0
        if config.debug_dir:
            Path(config.debug_dir).mkdir(parents=True, exist_ok=True)

    async def chat_text(self, messages: list[dict[str, str]], *, temperature: float | None = None, max_tokens: int | None = None, timeout: float | None = None, extra_body: dict | None = None) -> str:
        async with self._sem:
            if self.config.base_url.startswith("mock://"):
                return _mock_text(messages)
            return await asyncio.to_thread(self._chat_text_sync, messages, temperature, max_tokens, timeout, extra_body)

    async def chat_json(self, messages: list[dict[str, str]], *, schema_name: str, timeout: float | None = None, retries: int = 2, extra_body: dict | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(max(1, retries + 1)):
            text = await self.chat_text(messages, timeout=timeout, extra_body=extra_body)
            try:
                return _extract_json_object(text)
            except Exception as exc:
                last_error = exc
                messages = list(messages) + [{"role": "user", "content": f"Return valid JSON only for schema {schema_name}. Previous parse error: {exc}"}]
        raise ValueError(f"failed to parse JSON for {schema_name}: {last_error}")

    def _chat_text_sync(self, messages: list[dict[str, str]], temperature: float | None, max_tokens: int | None, timeout: float | None, extra_body: dict | None) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            from .openai_compat import OpenAI

        body: dict[str, Any] = dict(self.config.extra_body or {})
        if extra_body:
            body = _deep_merge(body, extra_body)
        if self.config.thinking_mode is not None:
            # Do not force disable thinking.  Keep this configurable, as required.
            ctk = body.setdefault("chat_template_kwargs", {})
            ctk.setdefault("enable_thinking", bool(self.config.thinking_mode))
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if body:
            kwargs["extra_body"] = body
        client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url, timeout=timeout or self.config.timeout_seconds)
        response = None
        wait_schedule = (0.0, *tuple(float(x) for x in self.config.retry_wait_seconds))
        for attempt, wait_seconds in enumerate(wait_schedule):
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            try:
                response = client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                retryable = _is_retryable_openai_error(exc)
                if attempt >= len(wait_schedule) - 1 or not retryable:
                    raise
                next_wait = wait_schedule[attempt + 1]
                status = _openai_status_code(exc)
                print(
                    "[dynamix-generation-retry] "
                    f"attempt={attempt + 1}/{len(wait_schedule)} "
                    f"next_wait_seconds={next_wait:g} "
                    f"error={type(exc).__name__} "
                    f"status={status if status is not None else 'unknown'} "
                    f"base_url={self.config.base_url} "
                    f"model={self.config.model}",
                    file=sys.stderr,
                    flush=True,
                )
        if response is None:
            raise RuntimeError("generation request failed without an exception")
        content = response.choices[0].message.content or ""
        self._write_debug(messages, content)
        return content

    def _write_debug(self, messages: list[dict[str, str]], content: str) -> None:
        if not self.config.debug_dir:
            return
        self._counter += 1
        path = Path(self.config.debug_dir) / f"generation_{self._counter:05d}.json"
        path.write_text(json.dumps({"messages": messages, "response": content}, ensure_ascii=False, indent=2), encoding="utf-8")


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._sem = asyncio.Semaphore(max(1, int(config.max_concurrency)))
        self._cache: _SqliteEmbeddingCache | None = None
        self.truncation_events: list[dict[str, Any]] = []
        if config.cache_path:
            self._cache = _SqliteEmbeddingCache(config.cache_path)

    async def embed_texts(self, texts: list[str], *, model: str | None = None, batch_size: int | None = None, cache_namespace: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        model_name = model or self.config.model
        namespace = cache_namespace or model_name
        prepared_pairs = [self._prepare_text(text, index=idx) for idx, text in enumerate(texts)]
        prepared_texts = [pair[0] for pair in prepared_pairs]
        results: list[list[float] | None] = [None] * len(prepared_texts)
        missing: list[tuple[int, str]] = []
        for idx, prepared_text in enumerate(prepared_texts):
            cached = self._cache.get(namespace, prepared_text) if self._cache else None
            if cached is not None:
                results[idx] = cached
            else:
                missing.append((idx, prepared_text))
        bs = int(batch_size or self.config.batch_size or 1)
        batches = [missing[offset: offset + bs] for offset in range(0, len(missing), bs)]

        async def run_batch(batch: list[tuple[int, str]]):
            batch_texts = [text for _, text in batch]
            vectors = await self._embed_uncached(batch_texts, model_name)
            return batch, vectors

        for batch, vectors in await asyncio.gather(*(run_batch(batch) for batch in batches)):
            for (idx, prepared_text), vector in zip(batch, vectors):
                results[idx] = vector
                if self._cache:
                    self._cache.set(namespace, prepared_text, vector)
        return [list(v or []) for v in results]

    def _prepare_text(self, text: str, *, index: int | None = None) -> tuple[str, dict[str, Any]]:
        max_tokens = max(1, int(self.config.effective_max_input_tokens))
        allow_fallback = self.config.base_url.startswith("mock://") or not bool(self.config.tokenizer_required)
        tokenizer_model = self.config.tokenizer_model or (None if self.config.base_url.startswith("mock://") else self.config.model)
        try:
            result = truncate_with_tokenizer(
                text,
                tokenizer_model=tokenizer_model,
                max_tokens=max_tokens,
                strategy=self.config.truncation_strategy,
                allow_regex_fallback=allow_fallback,
            )
        except TokenizerUnavailable:
            raise
        if not result.truncated:
            return text, {
                "truncated": False,
                "token_count": result.token_count,
                "max_input_tokens": max_tokens,
                "tokenizer": result.tokenizer_name,
                "strategy": result.strategy,
            }
        if not self.config.truncate_long_texts:
            raise ValueError(
                f"embedding input exceeds max_input_tokens={max_tokens}: "
                f"token_count={result.token_count}; set truncate_long_texts=true to truncate"
            )
        event = {
            "index": index,
            "original_chars": len(text),
            "truncated_chars": len(result.truncated_text),
            "token_count": result.token_count,
            "max_input_tokens": max_tokens,
            "tokenizer": result.tokenizer_name,
            "strategy": result.strategy,
        }
        self.truncation_events.append(event)
        return result.truncated_text, event

    def save_truncation_report(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model": self.config.model,
            "max_model_len": self.config.max_model_len,
            "max_input_tokens": self.config.effective_max_input_tokens,
            "truncate_long_texts": self.config.truncate_long_texts,
            "tokenizer_model": self.config.tokenizer_model or self.config.model,
            "tokenizer_required": self.config.tokenizer_required,
            "truncation_strategy": self.config.truncation_strategy,
            "event_count": len(self.truncation_events),
            "events": self.truncation_events,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _embed_uncached(self, texts: list[str], model_name: str) -> list[list[float]]:
        async with self._sem:
            if self.config.base_url.startswith("mock://"):
                return [_deterministic_embedding(text, dim=self.config.deterministic_dim) for text in texts]
            return await asyncio.to_thread(self._embed_uncached_sync, texts, model_name)

    def _embed_uncached_sync(self, texts: list[str], model_name: str) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError:
            from .openai_compat import OpenAI
        client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        response = client.embeddings.create(model=model_name, input=texts)
        return [list(item.embedding) for item in response.data]

    def close(self) -> None:
        if self._cache:
            self._cache.close()


class _SqliteEmbeddingCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS embeddings (namespace TEXT, key TEXT, vector TEXT, PRIMARY KEY(namespace, key))")
        self.conn.commit()

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, namespace: str, text: str) -> list[float] | None:
        cur = self.conn.execute("SELECT vector FROM embeddings WHERE namespace=? AND key=?", (namespace, self._key(text)))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def set(self, namespace: str, text: str, vector: list[float]) -> None:
        self.conn.execute("INSERT OR REPLACE INTO embeddings(namespace,key,vector) VALUES(?,?,?)", (namespace, self._key(text), json.dumps(vector)))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def _deterministic_embedding(text: str, *, dim: int) -> list[float]:
    vec = np.zeros(dim, dtype=float)
    tokens = [tok for tok in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in text).split() if tok]
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12:
        vec[0] = 1.0
        norm = 1.0
    return (vec / norm).astype(float).tolist()


def _mock_text(messages: list[dict[str, str]]) -> str:
    joined = "\n".join(m.get("content", "") for m in messages)
    if "Return valid JSON" in joined or "ExperienceCard" in joined or "cluster" in joined.lower():
        return json.dumps({
            "name": "Reusable spreadsheet procedure",
            "trigger": "Use when spreadsheet tasks require formula or cell/range manipulation.",
            "content": "Inspect the requested answer range, preserve workbook structure, write only the required output workbook, and verify the target cells after saving.",
            "placement": {
                "target": "skill_md",
                "reference_kind": "procedure",
                "file_slug": "reusable-spreadsheet-procedure",
                "rationale": "Broad and high-confidence procedure suitable for preload."
            },
            "confidence": 0.75
        }, ensure_ascii=False)
    return "ACTION: TASK_COMPLETE"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re_sub_fence(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start:end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found")


def re_sub_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _is_retryable_openai_error(exc: Exception) -> bool:
    status_code = _openai_status_code(exc)
    if status_code is not None:
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            return True
        if code == 400:
            return False
        return code == 408 or code == 409 or code == 429 or code >= 500
    return True


def _openai_status_code(exc: Exception) -> Any:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return status_code
    status_code = getattr(exc, "status", None)
    if status_code is not None:
        return status_code
    return getattr(exc, "code", None)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out

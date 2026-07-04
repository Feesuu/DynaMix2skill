from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import threading
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
    api_key_env_var: str | None = None
    temperature: float = 0.6
    timeout_seconds: float = 1200.0
    max_concurrency: int = 8
    thinking_mode: bool | None = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    debug_dir: str | None = None
    retry_wait_seconds: tuple[float, ...] = (2.0, 5.0, 15.0)

    @property
    def resolved_api_key(self) -> str:
        if self.api_key_env_var:
            return os.environ.get(self.api_key_env_var, self.api_key or "EMPTY")
        return self.api_key or "EMPTY"


@dataclass
class EmbeddingConfig:
    base_url: str = "mock://deterministic"
    model: str = "Qwen3-Embedding-8B"
    api_key: str = "EMPTY"
    api_key_env_var: str | None = None
    # Official handoff runs use the Qwen3-Embedding-8B service with a 32k input
    # window. Long trajectories are chunked upstream before this guard is hit.
    max_model_len: int = 32000
    max_input_tokens: int | None = 32000
    truncate_long_texts: bool = True
    tokenizer_model: str | None = None
    tokenizer_required: bool = True
    truncation_strategy: str = "head"
    batch_size: int = 8
    max_concurrency: int = 8
    cache_path: str | None = None
    deterministic_dim: int = 384

    @property
    def effective_max_input_tokens(self) -> int:
        return int(self.max_input_tokens or self.max_model_len)

    @property
    def resolved_api_key(self) -> str:
        if self.api_key_env_var:
            return os.environ.get(self.api_key_env_var, self.api_key or "EMPTY")
        return self.api_key or "EMPTY"


class GenerationClient:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self._sem = asyncio.Semaphore(max(1, int(config.max_concurrency)))
        self._counter = _existing_debug_max_index(config.debug_dir)
        self._debug_lock = threading.Lock()
        if config.debug_dir:
            Path(config.debug_dir).mkdir(parents=True, exist_ok=True)

    async def chat_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        extra_body: dict | None = None,
        response_format: dict[str, Any] | None = None,
        debug_metadata: dict[str, Any] | None = None,
    ) -> str:
        async with self._sem:
            body = self._request_extra_body(extra_body)
            debug_payload = self._debug_payload(messages, temperature, max_tokens, timeout, body, response_format, debug_metadata)
            cached = self._reuse_succeeded_debug_response(debug_payload)
            if cached is not None:
                _append_usage_record(
                    "DYNAMIX_GENERATION_USAGE_LOG",
                    {
                        "component": "dynamix_generation",
                        "client": "openai",
                        "model": self.config.model,
                        "endpoint": self.config.base_url,
                        "cache_hit": True,
                        "usage": {},
                        "request": {"message_count": len(messages)},
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
                return cached
            debug_path = self._start_debug(debug_payload)
            if self.config.base_url.startswith("mock://"):
                content = _mock_text(messages)
                self._finish_debug(debug_path, "succeeded", response=content)
                return content
            try:
                request_timeout = float(timeout or self.config.timeout_seconds)
                content = await self._chat_text_with_app_timeout(messages, temperature, max_tokens, timeout, body, response_format, request_timeout)
            except asyncio.TimeoutError as exc:
                error = TimeoutError(f"generation request exceeded timeout_seconds={request_timeout:g}")
                self._finish_debug(debug_path, "failed", error=error)
                raise error from exc
            except Exception as exc:
                self._finish_debug(debug_path, "failed", error=exc)
                raise
            self._finish_debug(debug_path, "succeeded", response=content)
            return content

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema_name: str,
        guided_json: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        retries: int = 2,
        extra_body: dict | None = None,
        debug_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        metadata = dict(debug_metadata or {})
        metadata.setdefault("schema_name", schema_name)
        request_extra_body = dict(extra_body or {})
        response_format = _json_schema_response_format(schema_name, guided_json) if guided_json is not None else None
        for attempt in range(max(1, retries + 1)):
            attempt_metadata = {**metadata, "json_parse_attempt": attempt + 1}
            text = await self.chat_text(
                messages,
                timeout=timeout,
                max_tokens=max_tokens,
                extra_body=request_extra_body,
                response_format=response_format,
                debug_metadata=attempt_metadata,
            )
            try:
                if guided_json is not None:
                    return _extract_strict_json_object(text)
                return _extract_json_object(text)
            except Exception as exc:
                last_error = exc
                messages = list(messages) + [{"role": "user", "content": f"Return valid JSON only for schema {schema_name}. Previous parse error: {exc}"}]
        raise ValueError(f"failed to parse JSON for {schema_name}: {last_error}")

    async def _chat_text_with_app_timeout(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        timeout: float | None,
        body: dict[str, Any],
        response_format: dict[str, Any] | None,
        request_timeout: float,
    ) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        def deliver_result(content: str) -> None:
            if not future.done():
                future.set_result(content)

        def deliver_exception(exc: BaseException) -> None:
            if not future.done():
                future.set_exception(exc)

        def worker() -> None:
            try:
                content = self._chat_text_sync(messages, temperature, max_tokens, timeout, body, response_format)
            except BaseException as exc:
                try:
                    loop.call_soon_threadsafe(deliver_exception, exc)
                except RuntimeError:
                    pass
            else:
                try:
                    loop.call_soon_threadsafe(deliver_result, content)
                except RuntimeError:
                    pass

        thread = threading.Thread(target=worker, name="dynamix-generation-request", daemon=True)
        thread.start()
        return await asyncio.wait_for(future, timeout=max(0.001, request_timeout))

    def _chat_text_sync(self, messages: list[dict[str, str]], temperature: float | None, max_tokens: int | None, timeout: float | None, body: dict[str, Any], response_format: dict[str, Any] | None) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            from .openai_compat import OpenAI

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if body:
            kwargs["extra_body"] = body
        if response_format:
            kwargs["response_format"] = response_format
        client = OpenAI(api_key=self.config.resolved_api_key, base_url=self.config.base_url, timeout=timeout or self.config.timeout_seconds)
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
        _append_usage_record(
            "DYNAMIX_GENERATION_USAGE_LOG",
            {
                "component": "dynamix_generation",
                "client": "openai",
                "model": self.config.model,
                "endpoint": self.config.base_url,
                "cache_hit": False,
                "usage": _response_usage_payload(response),
                "request": {
                    "message_count": len(messages),
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        content = response.choices[0].message.content or ""
        return content

    def _request_extra_body(self, extra_body: dict | None) -> dict[str, Any]:
        body: dict[str, Any] = dict(self.config.extra_body or {})
        if extra_body:
            body = _deep_merge(body, extra_body)
        if self.config.thinking_mode is not None:
            # Do not force disable thinking. Keep this configurable, as required.
            ctk = body.setdefault("chat_template_kwargs", {})
            ctk.setdefault("enable_thinking", bool(self.config.thinking_mode))
        return body

    def _debug_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        timeout: float | None,
        extra_body: dict[str, Any],
        response_format: dict[str, Any] | None,
        debug_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "status": "pending",
            "metadata": dict(debug_metadata or {}),
            "request": {
                "model": self.config.model,
                "base_url": self.config.base_url,
                "api_key": _api_key_fingerprint(self.config.resolved_api_key),
                "temperature": self.config.temperature if temperature is None else temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout or self.config.timeout_seconds,
                "extra_body": extra_body,
                "response_format": response_format,
            },
            "messages": messages,
        }

    def _reuse_succeeded_debug_response(self, payload: dict[str, Any]) -> str | None:
        if not self.config.debug_dir:
            return None
        debug_dir = Path(self.config.debug_dir)
        if not debug_dir.exists():
            return None
        expected = {
            "metadata": payload.get("metadata", {}),
            "request": payload.get("request", {}),
            "messages": payload.get("messages", []),
        }
        for path in sorted(debug_dir.glob("generation_*.json")):
            cached = _safe_read_debug_json(path)
            if not cached or cached.get("status") != "succeeded":
                continue
            if "response" not in cached:
                continue
            actual = {
                "metadata": cached.get("metadata", {}),
                "request": cached.get("request", {}),
                "messages": cached.get("messages", []),
            }
            if actual == expected:
                return str(cached.get("response") or "")
        return None

    def _start_debug(self, payload: dict[str, Any]) -> Path | None:
        if not self.config.debug_dir:
            return None
        while True:
            with self._debug_lock:
                self._counter += 1
                path = Path(self.config.debug_dir) / f"generation_{self._counter:05d}.json"
            created = _safe_write_debug_json_exclusive(path, payload)
            if created is True:
                return path
            if created is False:
                continue
            return None

    def _finish_debug(self, path: Path | None, status: str, *, response: str | None = None, error: Exception | None = None) -> None:
        if path is None:
            return
        payload = _safe_read_debug_json(path)
        if payload is None:
            payload = {}
        payload["status"] = status
        if response is not None:
            payload["response"] = response
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "status": _openai_status_code(error),
            }
        _safe_write_debug_json(path, payload)

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
        namespace = self._cache_namespace(cache_namespace or model_name, model_name=model_name)
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

    def _cache_namespace(self, logical_namespace: str, *, model_name: str) -> str:
        payload = {
            "base_url": self.config.base_url,
            "model": model_name,
            "api_key": _api_key_fingerprint(self.config.resolved_api_key),
            "max_model_len": int(self.config.max_model_len),
            "max_input_tokens": int(self.config.effective_max_input_tokens),
            "truncate_long_texts": bool(self.config.truncate_long_texts),
            "tokenizer_model": self.config.tokenizer_model or "",
            "tokenizer_required": bool(self.config.tokenizer_required),
            "truncation_strategy": self.config.truncation_strategy,
            "deterministic_dim": int(self.config.deterministic_dim),
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return f"{logical_namespace}::protocol::{digest}"

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
        client = OpenAI(api_key=self.config.resolved_api_key, base_url=self.config.base_url)
        response = client.embeddings.create(model=model_name, input=texts)
        _append_usage_record(
            "DYNAMIX_EMBEDDING_USAGE_LOG",
            {
                "component": "dynamix_embedding",
                "client": "openai_embeddings",
                "model": model_name,
                "endpoint": self.config.base_url,
                "cache_hit": False,
                "usage": _response_usage_payload(response),
                "request": {"input_count": len(texts)},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        return [list(item.embedding) for item in response.data]

    def close(self) -> None:
        if self._cache:
            self._cache.close()


def _safe_read_debug_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _debug_io_warning(path, "read", exc)
        return None
    return payload if isinstance(payload, dict) else None


def _existing_debug_max_index(debug_dir: str | None) -> int:
    if not debug_dir:
        return 0
    path = Path(debug_dir)
    if not path.exists():
        return 0
    max_index = 0
    for debug_file in path.glob("generation_*.json"):
        stem = debug_file.stem
        try:
            max_index = max(max_index, int(stem.removeprefix("generation_")))
        except ValueError:
            continue
    return max_index


def _safe_write_debug_json(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        _debug_io_warning(path, "write", exc)
        return False
    return True


def _safe_write_debug_json_exclusive(path: Path, payload: dict[str, Any]) -> bool | None:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    except Exception as exc:
        _debug_io_warning(path, "exclusive_create", exc)
        return None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        _debug_io_warning(path, "exclusive_write", exc)
        return None
    return True


def _debug_io_warning(path: Path, operation: str, exc: Exception) -> None:
    print(
        "[dynamix-generation-debug-warning] "
        f"operation={operation} "
        f"path={path} "
        f"error={type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )


def _api_key_fingerprint(value: str) -> str:
    if value == "EMPTY":
        return "EMPTY"
    if not value:
        return ""
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _response_usage_payload(value: Any) -> dict[str, Any]:
    usage = getattr(value, "usage", None)
    if usage is None and isinstance(value, dict):
        usage = value.get("usage")
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        return dict(usage.model_dump())
    if hasattr(usage, "dict"):
        return dict(usage.dict())
    payload: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        token_value = getattr(usage, key, None)
        if token_value is not None:
            payload[key] = token_value
    return payload


def _append_usage_record(env_var: str, payload: dict[str, Any]) -> None:
    path = os.getenv(env_var)
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception as exc:  # pragma: no cover - telemetry must not break builds
        print(
            "[dynamix-usage-warning] "
            f"path={path} error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


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
    text = _strip_think_blocks(text).strip()
    candidates = [text]
    candidates.extend(_fenced_code_blocks(text))
    if text.startswith("```"):
        candidates.append(re_sub_fence(text))
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    for candidate in candidates:
        for snippet in _balanced_json_object_candidates(candidate):
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
    raise ValueError("no JSON object found")


def _extract_strict_json_object(text: str) -> dict[str, Any]:
    text = _strip_think_blocks(text).strip()
    candidates = [text]
    if text.startswith("```") and text.endswith("```"):
        candidates.append(re_sub_fence(text))
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("guided JSON response was not a strict JSON object")


def _json_schema_response_format(schema_name: str, schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if schema is None:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


def _strip_think_blocks(text: str) -> str:
    out = str(text)
    while True:
        start = out.find("<think>")
        end = out.find("</think>", start + len("<think>")) if start >= 0 else -1
        if start < 0 or end < 0:
            return out
        out = out[:start] + out[end + len("</think>") :]


def _fenced_code_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines()
    in_block = False
    current: list[str] = []
    for line in lines:
        if line.strip().startswith("```"):
            if in_block:
                blocks.append("\n".join(current).strip())
                current = []
                in_block = False
            else:
                in_block = True
                current = []
            continue
        if in_block:
            current.append(line)
    return [block for block in blocks if block]


def _balanced_json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    return candidates


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

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from .openai_compat import OpenAI as CompatOpenAI


@dataclass(frozen=True)
class SkillNodeDocument:
    node_id: str
    item_id: str
    name: str
    trigger: str
    content: str
    embedding_text: str
    prompt_text: str
    sha256: str
    level: int = 0
    support_mass: float = 0.0
    confidence: float = 0.0
    source_community_id: str = ""
    source_member_count: int = 0
    analyst_mode: str = ""


@dataclass(frozen=True)
class SkillSelection:
    skill: SkillNodeDocument
    score: float


def discover_skill_documents(skillbank_root: str | Path) -> list[SkillNodeDocument]:
    root = Path(skillbank_root)
    if not root.exists():
        raise FileNotFoundError(root)
    node_manifest = root / "node_bank_manifest.json"
    if not node_manifest.exists():
        raise FileNotFoundError(f"node bank manifest not found: {node_manifest}")
    return _discover_node_documents(node_manifest)


def _discover_node_documents(manifest_path: Path) -> list[SkillNodeDocument]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != "dynamix_node_skill_bank_v1":
        raise ValueError(f"unsupported node bank manifest format: {payload.get('format')!r}")
    docs: list[SkillNodeDocument] = []
    for node in payload.get("nodes", []):
        node_id = str(node.get("node_id") or node.get("item_id") or "").strip()
        name = str(node.get("name") or node_id).strip()
        trigger = str(node.get("trigger") or "").strip()
        content = str(node.get("content") or "").strip()
        if not node_id or not name or not trigger or not content:
            continue
        embedding_text = str(node.get("embedding_text") or _render_node_embedding_text(name=name, trigger=trigger, content=content)).strip()
        prompt_text = str(node.get("prompt_text") or "").strip()
        sha256 = str(node.get("sha256") or hashlib.sha256(embedding_text.encode("utf-8")).hexdigest())
        docs.append(SkillNodeDocument(
            node_id=node_id,
            item_id=str(node.get("item_id") or node_id),
            name=name,
            trigger=trigger,
            content=content,
            embedding_text=embedding_text,
            prompt_text=prompt_text,
            sha256=sha256,
            level=int(node.get("level", 0) or 0),
            support_mass=float(node.get("support_mass", 0.0) or 0.0),
            confidence=float(node.get("confidence", 0.0) or 0.0),
            source_community_id=str(node.get("source_community_id", "") or ""),
            source_member_count=int(node.get("source_member_count", 0) or 0),
            analyst_mode=str(node.get("analyst_mode", "") or ""),
        ))
    if not docs:
        raise ValueError(f"no retrievable nodes found in node bank manifest: {manifest_path}")
    return docs


class SkillBankSelector:
    """Dense top-k selector over a DynaMix node bank.

    Each ExperienceCard node is embedded using only name, trigger, and content.
    A heldout task query is embedded once and ranked by cosine similarity.
    """

    def __init__(
        self,
        *,
        skillbank_root: str | Path,
        base_url: str = "mock://deterministic",
        model: str = "Qwen3-Embedding-8B",
        api_key: str = "EMPTY",
        cache_path: str | Path | None = None,
    ):
        self.skillbank_root = Path(skillbank_root)
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.cache_path = Path(cache_path) if cache_path else self.skillbank_root / ".dynamix_skillbank_index.json"
        self._docs: list[SkillNodeDocument] | None = None
        self._embeddings: np.ndarray | None = None

    @classmethod
    def from_env(cls, *, default_skillbank_root: str | Path | None = None) -> "SkillBankSelector":
        root = os.environ.get("DYNAMIX_SKILLBANK_ROOT") or (str(default_skillbank_root) if default_skillbank_root else "")
        if not root:
            raise ValueError("DYNAMIX_SKILLBANK_ROOT is required for skillbank selection")
        return cls(
            skillbank_root=root,
            base_url=os.environ.get("DYNAMIX_SKILLBANK_EMBED_BASE_URL", os.environ.get("EMBED_BASE_URL", "mock://deterministic")),
            model=os.environ.get("DYNAMIX_SKILLBANK_EMBED_MODEL", os.environ.get("EMBED_MODEL", "Qwen3-Embedding-8B")),
            api_key=os.environ.get("DYNAMIX_SKILLBANK_EMBED_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")),
            cache_path=os.environ.get("DYNAMIX_SKILLBANK_CACHE_PATH") or None,
        )

    def select(self, query_text: str, *, top_k: int = 3) -> list[SkillSelection]:
        docs, embeddings = self._load_or_build_index()
        query_embedding = np.asarray(self._embed([query_text])[0], dtype=float)
        query_embedding = _normalize(query_embedding)
        scores = embeddings @ query_embedding
        order = np.argsort(-scores)[: max(1, min(int(top_k), len(docs)))]
        return [SkillSelection(skill=docs[int(i)], score=float(scores[int(i)])) for i in order]

    def _load_or_build_index(self) -> tuple[list[SkillNodeDocument], np.ndarray]:
        if self._docs is not None and self._embeddings is not None:
            return self._docs, self._embeddings
        docs = discover_skill_documents(self.skillbank_root)
        expected = {doc.node_id: doc.sha256 for doc in docs}
        if self.cache_path.exists():
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if (
                    payload.get("model") == self.model
                    and payload.get("base_url") == self.base_url
                    and payload.get("api_key_fingerprint") == _api_key_fingerprint(self.api_key)
                    and payload.get("document_hashes") == expected
                ):
                    self._docs = [SkillNodeDocument(**item) for item in payload["documents"]]
                    self._embeddings = np.asarray(payload["embeddings"], dtype=float)
                    self._embeddings = _normalize_matrix(self._embeddings)
                    return self._docs, self._embeddings
            except Exception:
                pass
        embeddings = np.asarray(self._embed([doc.embedding_text for doc in docs]), dtype=float)
        embeddings = _normalize_matrix(embeddings)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "dynamix_skillbank_embedding_index_v2",
            "skillbank_root": str(self.skillbank_root),
            "model": self.model,
            "base_url": self.base_url,
            "api_key_fingerprint": _api_key_fingerprint(self.api_key),
            "document_hashes": expected,
            "documents": [asdict(doc) for doc in docs],
            "embeddings": embeddings.tolist(),
        }
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._docs = docs
        self._embeddings = embeddings
        return docs, embeddings

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.base_url.startswith("mock://"):
            return [_deterministic_embedding(text) for text in texts]
        client_cls = OpenAI or CompatOpenAI
        client = client_cls(api_key=self.api_key, base_url=self.base_url, timeout=600)
        response = client.embeddings.create(model=self.model, input=texts)
        _append_usage_record(
            "DYNAMIX_SKILLBANK_USAGE_LOG",
            {
                "component": "dynamix_skillbank_embedding",
                "client": "openai_embeddings",
                "model": self.model,
                "endpoint": self.base_url,
                "cache_hit": False,
                "usage": _response_usage_payload(response),
                "request": {"input_count": len(texts)},
                "timestamp": _utc_timestamp(),
            },
        )
        return [list(item.embedding) for item in response.data]


def selected_experience_to_system_content(selections: Iterable[SkillSelection]) -> str:
    selections = list(selections)
    lines: list[str] = [
        "# Retrieved Experience",
        "",
        "The following reusable experience was selected for this task. Use relevant guidance when it matches the spreadsheet operation; ignore irrelevant guidance.",
        "",
    ]
    for rank, selection in enumerate(selections, start=1):
        node = selection.skill
        lines.extend([
            f"## Node {rank}: {node.name}",
            "",
            f"Trigger: {node.trigger}",
            "",
            "Guidance:",
            node.content.strip(),
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


# No per-query copying helper is provided.  The nodebank is a run-level immutable
# directory; each query only selects top-k node records and injects their text
# into the prompt.  This is concurrency-safe for local multi-worker runs.


def _render_node_embedding_text(*, name: str, trigger: str, content: str) -> str:
    return "\n".join([
        f"name: {name.strip()}",
        f"trigger: {trigger.strip()}",
        f"content: {content.strip()}",
    ]).strip()


def _api_key_fingerprint(value: str) -> str:
    if value == "EMPTY":
        return "EMPTY"
    if not value:
        return ""
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_timestamp() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _response_usage_payload(value) -> dict:
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
    payload = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        token_value = getattr(usage, key, None)
        if token_value is not None:
            payload[key] = token_value
    return payload


def _append_usage_record(env_var: str, payload: dict) -> None:
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
    except Exception:
        return


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1.0e-12 else v


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 1.0e-12, 1.0, norms)
    return matrix / norms


def _deterministic_embedding(text: str, *, dim: int = 384) -> list[float]:
    vec = np.zeros(dim, dtype=float)
    for token in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    vec = _normalize(vec)
    return vec.astype(float).tolist()

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from .openai_compat import OpenAI as CompatOpenAI


@dataclass(frozen=True)
class SkillDocument:
    skill_id: str
    name: str
    description: str
    skill_dir: str
    skill_path: str
    content: str
    full_text: str
    sha256: str


@dataclass(frozen=True)
class SkillSelection:
    skill: SkillDocument
    score: float


def discover_skill_documents(skillbank_root: str | Path) -> list[SkillDocument]:
    root = Path(skillbank_root)
    docs: list[SkillDocument] = []
    if not root.exists():
        raise FileNotFoundError(root)
    for skill_path in sorted(root.rglob("SKILL.md")):
        # Avoid indexing nested copies inside selected-skill compatibility roots.
        if any(part in {"selected_skills", "__pycache__"} for part in skill_path.parts):
            continue
        raw = skill_path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(raw)
        name = str(frontmatter.get("name") or skill_path.parent.name).strip()
        description = str(frontmatter.get("description") or "").strip()
        skill_id = _stable_skill_id(skill_path.parent.name, raw)
        full_text = f"{name}\n{description}\n{body}".strip()
        docs.append(SkillDocument(
            skill_id=skill_id,
            name=name,
            description=description,
            skill_dir=str(skill_path.parent),
            skill_path=str(skill_path),
            content=body.strip(),
            full_text=full_text,
            sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        ))
    if not docs:
        raise ValueError(f"no SKILL.md files found under skillbank root: {root}")
    return docs


class SkillBankSelector:
    """Dense top-k skill selector over a folder of SKILL.md files.

    The selector embeds each SKILL.md once, embeds the current task query once,
    then ranks by cosine similarity.  This is the selection layer needed when a
    DynaMix run exports multiple skill folders but Trace2Skill can only preload a
    small task-conditioned subset into the system prompt.
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
        self._docs: list[SkillDocument] | None = None
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

    def _load_or_build_index(self) -> tuple[list[SkillDocument], np.ndarray]:
        if self._docs is not None and self._embeddings is not None:
            return self._docs, self._embeddings
        docs = discover_skill_documents(self.skillbank_root)
        expected = {doc.skill_path: doc.sha256 for doc in docs}
        if self.cache_path.exists():
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if payload.get("model") == self.model and payload.get("skill_hashes") == expected:
                    self._docs = [SkillDocument(**item) for item in payload["skills"]]
                    self._embeddings = np.asarray(payload["embeddings"], dtype=float)
                    self._embeddings = _normalize_matrix(self._embeddings)
                    return self._docs, self._embeddings
            except Exception:
                pass
        embeddings = np.asarray(self._embed([doc.full_text for doc in docs]), dtype=float)
        embeddings = _normalize_matrix(embeddings)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "dynamix_skillbank_embedding_index_v1",
            "skillbank_root": str(self.skillbank_root),
            "model": self.model,
            "base_url": self.base_url,
            "skill_hashes": expected,
            "skills": [asdict(doc) for doc in docs],
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
        return [list(item.embedding) for item in response.data]


def selected_skills_to_system_content(selections: Iterable[SkillSelection]) -> str:
    lines: list[str] = [
        "# Selected DynaMix Skills",
        "",
        "The following skills were selected for this task by dense embedding similarity between the task query and the skill bank. Follow relevant selected skill guidance as Trace2Skill would follow a preloaded SKILL.md.",
        "",
    ]
    for rank, selection in enumerate(selections, start=1):
        skill = selection.skill
        lines.extend([
            f"## Selected Skill {rank}: {skill.name}",
            "",
            f"- similarity_score: {selection.score:.6f}",
            f"- skill_directory: `{skill.skill_dir}`",
            f"- skill_file: `{skill.skill_path}`",
        ])
        if skill.description:
            lines.append(f"- description: {skill.description}")
        lines.extend(["", skill.content.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


# No per-query skill-folder copying helper is provided.  The skillbank is a
# run-level immutable directory.  Each query only selects top-k SKILL.md files
# and references their fixed absolute skill directories in the prompt.  This is
# concurrency-safe for local multi-worker runs.  If a future remote/Docker runner
# is used, stage or mount the whole skillbank root once before the run and rewrite
# the index paths to that remote-visible root; do not copy selected folders per task.


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    try:
        _, rest = text.split("---", 1)
        fm, body = rest.split("---", 1)
    except ValueError:
        return {}, text
    data: dict[str, str] = {}
    for line in fm.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip().strip('"\'')
    return data, body.lstrip("\n")


def _stable_skill_id(name: str, raw: str) -> str:
    return f"{_slugify(name, max_len=48)}--{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}"


def _slugify(text: str, max_len: int = 64) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return (text[:max_len].strip("-") or "skill")


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

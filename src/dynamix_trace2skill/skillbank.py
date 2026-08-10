from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from dynamix_core.certified_otd import select_budgeted_antichain
from dynamix_trace2skill.clients import (
    _SqliteEmbeddingCache,
    api_key_fingerprint,
    embedding_vector_sha256,
    ordered_embedding_vectors,
)
from dynamix_trace2skill.tokenization import get_tokenizer

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from .openai_compat import OpenAI as CompatOpenAI

_STRICT_SINGLE_VECTOR_TREE_POLICIES = frozenset(
    {
        "certified_dual_view_otd",
        "contract_cut_ebst",
        "evidence_balanced_skill_tree",
    }
)
_FULL_PROMPT_ANALYST_MODES = frozenset(
    {
        "evidence_bucket_consolidation",
        "cross_child_abstraction",
        "contract_cut_skill",
    }
)


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
    parent_node_id: str | None = None
    child_node_ids: tuple[str, ...] = ()
    token_cost: int = 1


@dataclass(frozen=True)
class SkillSelection:
    skill: SkillNodeDocument
    score: float


def retrieved_experience_preamble() -> str:
    return (
        "# Retrieved Experience\n\n"
        "The following reusable experience was selected for this task. "
        "Use relevant guidance when it matches the spreadsheet operation; "
        "ignore irrelevant guidance."
    )


def render_cdost_node_prompt(*, name: str, trigger: str, content: str) -> str:
    return f"### {name}\nTrigger: {trigger}\n{content}".strip()


def skillbank_vector_cache_namespace(
    *,
    base_url: str,
    model: str,
    api_key_fingerprint_value: str,
    embedding_protocol: Mapping[str, Any],
) -> str:
    namespace_payload = {
        "format": "dynamix_skillbank_single_vector_v1",
        "base_url": base_url,
        "model": model,
        "api_key": api_key_fingerprint_value,
        "embedding_protocol": dict(embedding_protocol),
    }
    return hashlib.sha256(
        json.dumps(
            namespace_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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
            parent_node_id=(
                str(node["parent_node_id"])
                if node.get("parent_node_id") is not None
                else None
            ),
            child_node_ids=tuple(
                str(child_id)
                for child_id in node.get("child_node_ids", [])
            ),
            token_cost=max(1, int(node.get("token_cost", 1) or 1)),
        ))
    if not docs:
        raise ValueError(f"no retrievable nodes found in node bank manifest: {manifest_path}")
    return docs


def _authoritative_cdost_tree(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> tuple[
    str,
    dict[str, tuple[str, ...]],
    set[str],
]:
    identity = manifest.get("authoritative_tree")
    if not isinstance(identity, dict):
        raise ValueError(
            "certified_dual_view_otd nodebank requires authoritative tree identity"
        )
    manifest_dir = manifest_path.parent.resolve()
    tree_root = manifest_dir.parent

    def load_payload(
        *,
        relative_key: str,
        sha_key: str,
        expected_name: str,
    ) -> dict[str, Any]:
        relative_value = identity.get(relative_key)
        reported_sha = str(identity.get(sha_key) or "").strip()
        relative_path = Path(str(relative_value or ""))
        if (
            not str(relative_value or "").strip()
            or relative_path.is_absolute()
            or not reported_sha
        ):
            raise ValueError(
                "certified_dual_view_otd authoritative tree identity is incomplete"
            )
        artifact_path = (manifest_dir / relative_path).resolve()
        if artifact_path.parent != tree_root or artifact_path.name != expected_name:
            raise ValueError(
                "certified_dual_view_otd authoritative tree path escapes its build"
            )
        payload_bytes = artifact_path.read_bytes()
        if hashlib.sha256(payload_bytes).hexdigest() != reported_sha:
            raise ValueError(
                "certified_dual_view_otd authoritative tree digest mismatch"
            )
        payload = json.loads(payload_bytes.decode("utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("format") != "certified_dual_view_otd_v1"
        ):
            raise ValueError(
                "certified_dual_view_otd authoritative tree format is invalid"
            )
        return payload

    state_payload = load_payload(
        relative_key="state_relative_path",
        sha_key="state_sha256",
        expected_name="otd_tree_state.json",
    )
    structure_payload = load_payload(
        relative_key="structure_relative_path",
        sha_key="structure_sha256",
        expected_name="otd_tree_structure.json",
    )

    def topology(
        payload: Mapping[str, Any],
    ) -> tuple[str, dict[str, tuple[str, ...]]]:
        root_node_id = str(payload.get("root_node_id") or "").strip()
        raw_nodes = payload.get("nodes")
        if not root_node_id or not isinstance(raw_nodes, dict):
            raise ValueError(
                "certified_dual_view_otd authoritative tree is incomplete"
            )
        children_by_node: dict[str, tuple[str, ...]] = {}
        for raw_node_id, raw_node in raw_nodes.items():
            node_id = str(raw_node_id).strip()
            if (
                not node_id
                or not isinstance(raw_node, dict)
                or str(raw_node.get("node_id") or "").strip() != node_id
            ):
                raise ValueError(
                    "certified_dual_view_otd authoritative node is malformed"
                )
            left_id = raw_node.get("left_id")
            right_id = raw_node.get("right_id")
            if (left_id is None) != (right_id is None):
                raise ValueError(
                    "certified_dual_view_otd authoritative tree is not binary"
                )
            if left_id is None:
                if raw_node.get("atom_id") is None:
                    raise ValueError(
                        "certified_dual_view_otd authoritative leaf has no atom"
                    )
                child_ids: tuple[str, ...] = ()
            else:
                if raw_node.get("atom_id") is not None:
                    raise ValueError(
                        "certified_dual_view_otd authoritative internal node has an atom"
                    )
                child_ids = (str(left_id).strip(), str(right_id).strip())
                if any(not child_id for child_id in child_ids):
                    raise ValueError(
                        "certified_dual_view_otd authoritative child is missing"
                    )
            children_by_node[node_id] = child_ids
        return root_node_id, children_by_node

    state_root, state_children = topology(state_payload)
    structure_root, structure_children = topology(structure_payload)
    if (
        state_root != structure_root
        or state_children != structure_children
    ):
        raise ValueError(
            "certified_dual_view_otd state and structure topology disagree"
        )
    if int(identity.get("structural_node_count", -1)) != len(state_children):
        raise ValueError(
            "certified_dual_view_otd authoritative structural count mismatch"
        )

    retrievable_node_ids: set[str] = set()
    for node_id, raw_node in state_payload["nodes"].items():
        if not bool(raw_node.get("retrievable")):
            continue
        skill = raw_node.get("skill")
        if not isinstance(skill, dict) or any(
            not str(skill.get(field_name) or "").strip()
            for field_name in ("name", "trigger", "content")
        ):
            raise ValueError(
                "certified_dual_view_otd authoritative retrievable node is incomplete"
            )
        retrievable_node_ids.add(str(node_id))
    if int(identity.get("retrievable_node_count", -1)) != len(
        retrievable_node_ids
    ):
        raise ValueError(
            "certified_dual_view_otd authoritative retrievable count mismatch"
        )
    return state_root, state_children, retrievable_node_ids


def _validate_cdost_tree_index(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> tuple[str, dict[str, tuple[str, ...]]]:
    tree_index = manifest.get("tree_index")
    if not isinstance(tree_index, dict):
        raise ValueError(
            "certified_dual_view_otd nodebank requires a complete tree_index"
        )
    root_node_id = str(tree_index.get("root_node_id") or "").strip()
    raw_children = tree_index.get("children_by_node")
    if not root_node_id or not isinstance(raw_children, dict):
        raise ValueError(
            "certified_dual_view_otd nodebank requires a complete tree_index"
        )

    children_by_node: dict[str, tuple[str, ...]] = {}
    for raw_node_id, raw_child_ids in raw_children.items():
        node_id = str(raw_node_id).strip()
        if not node_id or not isinstance(raw_child_ids, list):
            raise ValueError("certified_dual_view_otd tree_index is malformed")
        child_ids = tuple(str(child_id).strip() for child_id in raw_child_ids)
        if (
            any(not child_id for child_id in child_ids)
            or len(child_ids) not in {0, 2}
            or len(child_ids) != len(set(child_ids))
        ):
            raise ValueError(
                "certified_dual_view_otd tree_index must be binary without "
                "duplicate edges"
            )
        children_by_node[node_id] = child_ids

    node_ids = set(children_by_node)
    if root_node_id not in node_ids:
        raise ValueError("certified_dual_view_otd tree_index root is missing")
    parent_count = {node_id: 0 for node_id in node_ids}
    for child_ids in children_by_node.values():
        for child_id in child_ids:
            if child_id not in node_ids:
                raise ValueError(
                    "certified_dual_view_otd tree_index is not closed"
                )
            parent_count[child_id] += 1
    if parent_count[root_node_id] != 0 or any(
        count != 1
        for node_id, count in parent_count.items()
        if node_id != root_node_id
    ):
        raise ValueError(
            "certified_dual_view_otd tree_index must have unique parents"
        )
    reported_root_node_id = str(manifest.get("root_node_id") or "").strip()
    if reported_root_node_id != root_node_id:
        raise ValueError(
            "certified_dual_view_otd manifest root does not match tree_index"
        )
    parent_by_node = {
        child_id: node_id
        for node_id, child_ids in children_by_node.items()
        for child_id in child_ids
    }

    visiting: set[str] = set()
    visited: set[str] = set()
    stack = [(root_node_id, False)]
    while stack:
        node_id, exiting = stack.pop()
        if exiting:
            visiting.remove(node_id)
            visited.add(node_id)
            continue
        if node_id in visiting:
            raise ValueError("certified_dual_view_otd tree_index contains a cycle")
        if node_id in visited:
            continue
        visiting.add(node_id)
        stack.append((node_id, True))
        stack.extend(
            (child_id, False)
            for child_id in reversed(children_by_node[node_id])
        )
    if visited != node_ids:
        raise ValueError(
            "certified_dual_view_otd tree_index contains unreachable nodes"
        )

    manifest_node_ids: list[str] = []
    for node in manifest.get("nodes", []):
        if not isinstance(node, dict):
            raise ValueError("certified_dual_view_otd node entry is malformed")
        node_id = str(node.get("node_id") or node.get("item_id") or "").strip()
        if not node_id or any(
            not str(node.get(field_name) or "").strip()
            for field_name in ("name", "trigger", "content")
        ):
            raise ValueError("certified_dual_view_otd node entry is incomplete")
        if "parent_node_id" not in node or not isinstance(
            node.get("child_node_ids"),
            list,
        ):
            raise ValueError(
                "certified_dual_view_otd node lineage is incomplete"
            )
        reported_parent = node.get("parent_node_id")
        reported_parent_id = (
            str(reported_parent).strip()
            if reported_parent is not None
            else None
        )
        expected_parent_id = parent_by_node.get(node_id)
        reported_children = tuple(
            str(child_id).strip()
            for child_id in node["child_node_ids"]
        )
        if (
            reported_parent_id != expected_parent_id
            or reported_children != children_by_node.get(node_id)
        ):
            raise ValueError(
                "certified_dual_view_otd node lineage does not match tree_index"
            )
        manifest_node_ids.append(node_id)
    if int(manifest.get("node_count", -1)) != len(manifest_node_ids):
        raise ValueError(
            "certified_dual_view_otd node_count does not match retrievable nodes"
        )
    if (
        len(manifest_node_ids) != len(set(manifest_node_ids))
        or not set(manifest_node_ids).issubset(node_ids)
    ):
        raise ValueError(
            "certified_dual_view_otd retrievable nodes do not match tree_index"
        )
    leaf_node_ids = {
        node_id
        for node_id, child_ids in children_by_node.items()
        if not child_ids
    }
    if not leaf_node_ids.issubset(manifest_node_ids):
        raise ValueError(
            "certified_dual_view_otd nodebank omits retrievable atom leaves"
        )

    (
        authoritative_root_node_id,
        authoritative_children_by_node,
        authoritative_retrievable_node_ids,
    ) = _authoritative_cdost_tree(
        manifest,
        manifest_path=manifest_path,
    )
    if (
        root_node_id != authoritative_root_node_id
        or children_by_node != authoritative_children_by_node
    ):
        raise ValueError(
            "certified_dual_view_otd manifest topology does not match "
            "authoritative tree"
        )
    if set(manifest_node_ids) != authoritative_retrievable_node_ids:
        raise ValueError(
            "certified_dual_view_otd retrievable nodes do not match "
            "authoritative tree"
        )
    return root_node_id, children_by_node


def _validate_ebst_tree_index(
    manifest: dict[str, Any],
) -> tuple[str, dict[str, tuple[str, ...]]]:
    label = "evidence_balanced_skill_tree"
    tree_index = manifest.get("tree_index")
    if not isinstance(tree_index, dict):
        raise ValueError(f"{label} nodebank requires a complete tree_index")
    root_node_id = str(tree_index.get("root_node_id") or "").strip()
    raw_children = tree_index.get("children_by_node")
    if not root_node_id or not isinstance(raw_children, dict):
        raise ValueError(f"{label} nodebank requires a complete tree_index")

    children_by_node: dict[str, tuple[str, ...]] = {}
    for raw_node_id, raw_child_ids in raw_children.items():
        node_id = str(raw_node_id).strip()
        if not node_id or not isinstance(raw_child_ids, list):
            raise ValueError(f"{label} tree_index is malformed")
        child_ids = tuple(str(child_id).strip() for child_id in raw_child_ids)
        if (
            any(not child_id for child_id in child_ids)
            or len(child_ids) != len(set(child_ids))
        ):
            raise ValueError(
                f"{label} tree_index contains an empty or duplicate edge"
            )
        children_by_node[node_id] = child_ids

    node_ids = set(children_by_node)
    if root_node_id not in node_ids:
        raise ValueError(f"{label} tree_index root is missing")
    reported_root = str(manifest.get("root_node_id") or "").strip()
    if reported_root != root_node_id:
        raise ValueError(f"{label} manifest root does not match tree_index")

    parent_by_node: dict[str, str] = {}
    for parent_id, child_ids in children_by_node.items():
        for child_id in child_ids:
            if child_id not in node_ids:
                raise ValueError(f"{label} tree_index is not closed")
            if child_id == root_node_id or child_id in parent_by_node:
                raise ValueError(f"{label} tree_index must have unique parents")
            parent_by_node[child_id] = parent_id
    if set(parent_by_node) != node_ids - {root_node_id}:
        raise ValueError(f"{label} tree_index contains unreachable nodes")

    visited: set[str] = set()
    stack = [root_node_id]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            raise ValueError(f"{label} tree_index contains a cycle")
        visited.add(node_id)
        stack.extend(children_by_node[node_id])
    if visited != node_ids:
        raise ValueError(f"{label} tree_index contains unreachable nodes")

    manifest_nodes = manifest.get("nodes")
    if not isinstance(manifest_nodes, list):
        raise ValueError(f"{label} node entries are missing")
    manifest_node_ids: list[str] = []
    for node in manifest_nodes:
        if not isinstance(node, dict):
            raise ValueError(f"{label} node entry is malformed")
        node_id = str(node.get("node_id") or node.get("item_id") or "").strip()
        if (
            not node_id
            or "parent_node_id" not in node
            or not isinstance(node.get("child_node_ids"), list)
        ):
            raise ValueError(f"{label} node lineage is incomplete")
        name = str(node.get("name") or "").strip()
        trigger = str(node.get("trigger") or "").strip()
        content = str(node.get("content") or "").strip()
        prompt_text = str(node.get("prompt_text") or "").strip()
        analyst_mode = str(node.get("analyst_mode") or "").strip()
        evidence_atom_ids = node.get("evidence_atom_ids")
        if (
            not name
            or not trigger
            or not content
            or not prompt_text
            or node.get("lifecycle_status") != "active"
            or analyst_mode not in _FULL_PROMPT_ANALYST_MODES
            or not isinstance(evidence_atom_ids, list)
            or len(evidence_atom_ids) < 2
            or len(evidence_atom_ids) != len(set(evidence_atom_ids))
            or any(not str(atom_id).strip() for atom_id in evidence_atom_ids)
        ):
            raise ValueError(f"{label} node entry is not retrievable")
        expected_embedding_text = (
            f"name: {name}\n"
            f"trigger: {trigger}\n"
            f"content: {content}"
        )
        if str(node.get("embedding_text") or "") != expected_embedding_text:
            raise ValueError(
                f"{label} node embedding contract does not match"
            )
        expected_parent = parent_by_node.get(node_id)
        reported_parent = str(node.get("parent_node_id") or "").strip() or None
        reported_children = tuple(
            str(child_id).strip()
            for child_id in node["child_node_ids"]
        )
        if (
            reported_parent != expected_parent
            or reported_children != children_by_node.get(node_id)
        ):
            raise ValueError(
                f"{label} node lineage does not match tree_index"
            )
        manifest_node_ids.append(node_id)

    expected_manifest_ids = node_ids - {root_node_id}
    if (
        len(manifest_node_ids) != len(set(manifest_node_ids))
        or set(manifest_node_ids) != expected_manifest_ids
        or int(manifest.get("node_count", -1)) != len(manifest_node_ids)
    ):
        raise ValueError(
            f"{label} retrievable nodes do not match tree_index"
        )
    return root_node_id, children_by_node


def validate_ebst_nodebank_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, dict[str, tuple[str, ...]]]:
    if manifest.get("tree_policy") != "evidence_balanced_skill_tree":
        raise ValueError(
            "expected evidence_balanced_skill_tree nodebank manifest"
        )
    return _validate_ebst_tree_index(dict(manifest))


class SkillBankSelector:
    """Policy-aware selector over a DynaMix node bank.

    Each ExperienceCard node is embedded using only name, trigger, and content.
    Existing manifests use dense cosine top-k; CDOST manifests use their
    declared tree-antichain objective.
    """

    def __init__(
        self,
        *,
        skillbank_root: str | Path,
        base_url: str = "mock://deterministic",
        model: str = "Qwen3-Embedding-8B",
        api_key: str = "EMPTY",
        cache_path: str | Path | None = None,
        vector_cache_path: str | Path | None = None,
        require_vector_cache_match: bool = False,
        max_model_len: int = 32000,
        max_input_tokens: int | None = 32000,
        batch_size: int = 8,
        tokenizer_model: str | None = None,
        chunk_tokens: int | None = None,
        chunk_overlap_tokens: int | None = None,
        require_cache_match: bool = False,
        expected_tree_policy: str | None = None,
    ):
        self.skillbank_root = Path(skillbank_root)
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.cache_path = Path(cache_path) if cache_path else self.skillbank_root / ".dynamix_skillbank_index.json"
        self.vector_cache_path = (
            Path(vector_cache_path) if vector_cache_path else None
        )
        self.require_vector_cache_match = bool(require_vector_cache_match)
        if self.require_vector_cache_match and self.vector_cache_path is None:
            raise ValueError(
                "strict skillbank vector reuse requires vector_cache_path"
            )
        self.max_model_len = int(max_model_len)
        self.max_input_tokens = int(max_input_tokens or max_model_len)
        if self.max_model_len <= 0 or self.max_input_tokens <= 0:
            raise ValueError("skillbank embedding token limits must be positive")
        self.batch_size = max(1, int(batch_size or 1))
        self.tokenizer_model = tokenizer_model or ""
        self.chunk_tokens = int(chunk_tokens) if chunk_tokens is not None else None
        self.chunk_overlap_tokens = int(chunk_overlap_tokens) if chunk_overlap_tokens is not None else None
        self.require_cache_match = bool(require_cache_match)
        self.expected_tree_policy = str(expected_tree_policy or "").strip()
        self._docs: list[SkillNodeDocument] | None = None
        self._embeddings: np.ndarray | None = None
        self.last_embedding_audit: dict[str, Any] = {}

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
            vector_cache_path=(
                os.environ.get("DYNAMIX_SKILLBANK_VECTOR_CACHE_PATH") or None
            ),
            require_vector_cache_match=os.environ.get(
                "DYNAMIX_SKILLBANK_REQUIRE_VECTOR_CACHE_MATCH",
                "",
            ).strip().casefold()
            in {"1", "true", "yes", "on"},
            max_model_len=int(os.environ.get("DYNAMIX_SKILLBANK_EMBED_MAX_MODEL_LEN", "32000")),
            max_input_tokens=int(os.environ.get("DYNAMIX_SKILLBANK_EMBED_MAX_INPUT_TOKENS", "32000")),
            batch_size=int(os.environ.get("DYNAMIX_SKILLBANK_EMBED_BATCH_SIZE", "8")),
            tokenizer_model=os.environ.get("DYNAMIX_SKILLBANK_EMBED_TOKENIZER") or None,
            chunk_tokens=int(os.environ["DYNAMIX_SKILLBANK_CHUNK_TOKENS"]) if os.environ.get("DYNAMIX_SKILLBANK_CHUNK_TOKENS") else None,
            chunk_overlap_tokens=int(os.environ["DYNAMIX_SKILLBANK_CHUNK_OVERLAP_TOKENS"]) if os.environ.get("DYNAMIX_SKILLBANK_CHUNK_OVERLAP_TOKENS") else None,
            require_cache_match=os.environ.get(
                "DYNAMIX_SKILLBANK_REQUIRE_CACHE_MATCH",
                "",
            ).strip().casefold()
            in {"1", "true", "yes", "on"},
            expected_tree_policy=(
                os.environ.get("DYNAMIX_SKILLBANK_EXPECT_TREE_POLICY") or None
            ),
        )

    def select(self, query_text: str, *, top_k: int = 3) -> list[SkillSelection]:
        manifest_path = self.skillbank_root / "node_bank_manifest.json"
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        export_policy = dict(manifest.get("export_policy", {}))
        reported_tree_policy = str(manifest.get("tree_policy") or "").strip()
        if (
            self.expected_tree_policy
            and reported_tree_policy != self.expected_tree_policy
        ):
            raise ValueError(
                "nodebank tree_policy does not match the expected experiment "
                "policy"
            )
        is_cdost = (
            reported_tree_policy == "certified_dual_view_otd"
            or self.expected_tree_policy == "certified_dual_view_otd"
        )
        if is_cdost and not self.expected_tree_policy:
            self.expected_tree_policy = "certified_dual_view_otd"
        is_ebst = (
            reported_tree_policy == "evidence_balanced_skill_tree"
            or self.expected_tree_policy == "evidence_balanced_skill_tree"
        )
        if is_ebst and not self.expected_tree_policy:
            self.expected_tree_policy = "evidence_balanced_skill_tree"
        is_contract_cut = (
            reported_tree_policy == "contract_cut_ebst"
            or self.expected_tree_policy == "contract_cut_ebst"
        )
        if is_contract_cut and not self.expected_tree_policy:
            self.expected_tree_policy = "contract_cut_ebst"
        if (
            (is_cdost or is_ebst)
            and export_policy.get("heldout_retrieval")
            != "tree_antichain_knapsack"
        ):
            raise ValueError(
                "single-parent nodebank requires "
                "tree_antichain_knapsack retrieval"
            )
        antichain_retrieval = (
            export_policy.get("heldout_retrieval")
            == "tree_antichain_knapsack"
        )
        tree_index = dict(manifest.get("tree_index", {}))
        if is_cdost:
            root_node_id, children_by_node = _validate_cdost_tree_index(
                manifest,
                manifest_path=manifest_path,
            )
            tree_index = {
                "root_node_id": root_node_id,
                "children_by_node": children_by_node,
            }
        elif is_ebst:
            root_node_id, children_by_node = validate_ebst_nodebank_manifest(
                manifest
            )
            tree_index = {
                "root_node_id": root_node_id,
                "children_by_node": children_by_node,
            }
        docs, embeddings = self._load_or_build_index()
        query_embedding = np.asarray(self._embed([query_text])[0], dtype=float)
        query_embedding = _normalize(query_embedding)
        self.last_embedding_audit["scoring_vector_sha256"] = [
            embedding_vector_sha256(query_embedding.tolist())
        ]
        self.last_embedding_audit["scoring_vector_policy"] = (
            "l2_normalized_query_used_by_dot_product"
        )
        scores = embeddings @ query_embedding
        if antichain_retrieval:
            score_by_id = {
                doc.node_id: max(0.0, min(1.0, (1.0 + float(score)) / 2.0))
                for doc, score in zip(docs, scores)
            }
            by_id = {doc.node_id: doc for doc in docs}
            retrieval_token_budget = int(export_policy["token_budget"])
            prompt_tokenizer = dict(export_policy.get("tokenizer", {}))
            prompt_tokenizer_model = str(
                prompt_tokenizer.get("model_or_path") or ""
            ).strip()
            tokenizer = get_tokenizer(
                prompt_tokenizer_model or None,
                allow_regex_fallback=bool(
                    prompt_tokenizer.get(
                        "regex_fallback_allowed",
                        False,
                    )
                ),
            )

            def selections_for_ids(
                node_ids: tuple[str, ...],
            ) -> list[SkillSelection]:
                return sorted(
                    (
                        SkillSelection(
                            skill=by_id[node_id],
                            score=score_by_id[node_id],
                        )
                        for node_id in node_ids
                    ),
                    key=lambda selection: (
                        -selection.score,
                        selection.skill.node_id,
                    ),
                )

            def exact_rendered_cost(node_ids: tuple[str, ...]) -> int:
                if not node_ids:
                    return 0
                return tokenizer.count(
                    selected_experience_to_system_content(
                        selections_for_ids(node_ids)
                    )
                )

            result = select_budgeted_antichain(
                root_node_id=str(tree_index["root_node_id"]),
                children_by_node={
                    str(node_id): tuple(str(child) for child in children)
                    for node_id, children in dict(
                        tree_index.get("children_by_node", {})
                    ).items()
                },
                relevance_by_node=score_by_id,
                token_cost_by_node={
                    doc.node_id: doc.token_cost
                    for doc in docs
                },
                max_nodes=max(1, int(top_k)),
                token_budget=retrieval_token_budget,
                token_unit=int(export_policy.get("token_unit", 128)),
                total_cost_for_nodes=exact_rendered_cost,
                max_exact_states=int(
                    export_policy.get("exact_search_max_states", 250_000)
                ),
            )
            selected = selections_for_ids(result.node_ids)
            if selected:
                rendered_tokens = tokenizer.count(
                    selected_experience_to_system_content(selected)
                )
                if rendered_tokens > retrieval_token_budget:
                    raise RuntimeError(
                        "selected experience exceeds the configured retrieval "
                        f"token budget: {rendered_tokens} > "
                        f"{retrieval_token_budget}"
                    )
            return selected
        order = np.argsort(-scores)[: max(1, min(int(top_k), len(docs)))]
        return [SkillSelection(skill=docs[int(i)], score=float(scores[int(i)])) for i in order]

    def prepare_index(self) -> dict[str, Any]:
        """Materialize and validate the immutable document embedding index."""
        docs, embeddings = self._load_or_build_index()
        return {
            "document_count": len(docs),
            "embedding_dimension": (
                int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
            ),
            "cache_path": str(self.cache_path.resolve()),
        }

    def _load_or_build_index(self) -> tuple[list[SkillNodeDocument], np.ndarray]:
        if self._docs is not None and self._embeddings is not None:
            return self._docs, self._embeddings
        docs = discover_skill_documents(self.skillbank_root)
        expected = {
            doc.node_id: hashlib.sha256(
                doc.embedding_text.encode("utf-8")
            ).hexdigest()
            for doc in docs
        }
        mismatched_manifest_hashes = [
            doc.node_id
            for doc in docs
            if doc.sha256 != expected[doc.node_id]
        ]
        if self.require_cache_match and mismatched_manifest_hashes:
            raise RuntimeError(
                "strict skillbank manifest document hash mismatch for "
                f"{mismatched_manifest_hashes[:10]}"
            )
        if self.cache_path.exists():
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                cache_matches = (
                    payload.get("format")
                    == "dynamix_skillbank_embedding_index_v2"
                    and payload.get("model") == self.model
                    and payload.get("base_url") == self.base_url
                    and payload.get("api_key_fingerprint") == api_key_fingerprint(self.api_key)
                    and self._embedding_protocol_matches(
                        payload.get("embedding_protocol")
                    )
                    and payload.get("document_hashes") == expected
                )
                if cache_matches:
                    cached_documents = [
                        SkillNodeDocument(**raw)
                        for raw in payload["documents"]
                    ]
                    cached_embeddings = np.asarray(
                        payload["embeddings"],
                        dtype=float,
                    )
                    if len(cached_documents) != len(cached_embeddings):
                        raise ValueError(
                            "cached document and embedding counts differ"
                        )
                    cached_row_by_id = {
                        document.node_id: index
                        for index, document in enumerate(cached_documents)
                    }
                    if len(cached_row_by_id) != len(cached_documents):
                        raise ValueError("cached node ids are not unique")
                    if set(cached_row_by_id) != set(expected):
                        raise ValueError("cached node ids do not match manifest")
                    self._docs = docs
                    self._embeddings = np.asarray(
                        [
                            cached_embeddings[cached_row_by_id[doc.node_id]]
                            for doc in docs
                        ],
                        dtype=float,
                    )
                    if (
                        self.expected_tree_policy
                        in _STRICT_SINGLE_VECTOR_TREE_POLICIES
                    ):
                        self._embeddings = _require_unit_matrix(
                            self._embeddings,
                            field_name="cached CDOST skillbank embeddings",
                        )
                    else:
                        self._embeddings = _normalize_matrix(self._embeddings)
                    return self._docs, self._embeddings
                if self.require_cache_match:
                    raise ValueError(
                        "skillbank cache does not match the heldout embedding protocol"
                    )
            except Exception as exc:
                if self.require_cache_match:
                    raise RuntimeError(
                        f"strict skillbank cache validation failed: {self.cache_path}"
                    ) from exc
        elif self.require_cache_match:
            raise FileNotFoundError(
                f"strict skillbank cache is missing: {self.cache_path}"
            )
        embeddings = np.asarray(self._embed([doc.embedding_text for doc in docs]), dtype=float)
        embeddings = _normalize_matrix(embeddings)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "dynamix_skillbank_embedding_index_v2",
            "skillbank_root": str(self.skillbank_root),
            "model": self.model,
            "base_url": self.base_url,
            "api_key_fingerprint": api_key_fingerprint(self.api_key),
            "embedding_protocol": self._embedding_protocol_payload(),
            "document_hashes": expected,
            "documents": [asdict(doc) for doc in docs],
            "embeddings": embeddings.tolist(),
        }
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._docs = docs
        self._embeddings = embeddings
        return docs, embeddings

    def _embed(self, texts: list[str]) -> list[list[float]]:
        strict_single_vector = (
            self.expected_tree_policy
            in _STRICT_SINGLE_VECTOR_TREE_POLICIES
        )
        if strict_single_vector:
            tokenizer = get_tokenizer(
                self.tokenizer_model or None,
                allow_regex_fallback=not bool(self.tokenizer_model),
            )
            token_limit = min(self.max_model_len, self.max_input_tokens)
            over_limit: list[tuple[int, int]] = []
            for index, text in enumerate(texts):
                token_count = tokenizer.count(text)
                if token_count > token_limit:
                    over_limit.append((index, token_count))
            if over_limit:
                index, token_count = over_limit[0]
                raise ValueError(
                    "skillbank single-vector embedding input exceeds its token "
                    f"limit: index={index}, tokens={token_count}, "
                    f"limit={token_limit}"
                )
        namespace = skillbank_vector_cache_namespace(
            base_url=self.base_url,
            model=self.model,
            api_key_fingerprint_value=api_key_fingerprint(self.api_key),
            embedding_protocol=self._embedding_protocol_payload(),
        )
        vector_cache = (
            _SqliteEmbeddingCache(
                self.vector_cache_path,
                write_policy="first_write_wins",
            )
            if self.vector_cache_path
            else None
        )
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        try:
            for index, text in enumerate(texts):
                cached = (
                    vector_cache.get(namespace, text)
                    if vector_cache is not None
                    else None
                )
                if cached is None:
                    missing.append((index, text))
                else:
                    results[index] = cached
            if self.require_vector_cache_match and missing:
                raise RuntimeError(
                    "strict skillbank vector cache is missing "
                    f"{len(missing)} required text embeddings"
                )

            if self.base_url.startswith("mock://"):
                generated_batches = [
                    (
                        missing,
                        [_deterministic_embedding(text) for _, text in missing],
                    )
                ]
            else:
                client_cls = OpenAI or CompatOpenAI
                client = client_cls(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=600,
                )
                generated_batches = []
                for offset in range(0, len(missing), self.batch_size):
                    indexed_batch = missing[offset: offset + self.batch_size]
                    batch = [text for _, text in indexed_batch]
                    response = client.embeddings.create(
                        model=self.model,
                        input=batch,
                    )
                    _append_usage_record(
                        "DYNAMIX_SKILLBANK_USAGE_LOG",
                        {
                            "component": "dynamix_skillbank_embedding",
                            "client": "openai_embeddings",
                            "model": self.model,
                            "endpoint": self.base_url,
                            "cache_hit": False,
                            "usage": _response_usage_payload(response),
                            "request": {
                                "input_count": len(batch),
                                "batch_size": self.batch_size,
                                "embedding_protocol": (
                                    self._embedding_protocol_payload()
                                ),
                            },
                            "timestamp": _utc_timestamp(),
                        },
                    )
                    generated_batches.append(
                        (
                            indexed_batch,
                            ordered_embedding_vectors(
                                response.data,
                                expected_count=len(batch),
                            ),
                        )
                    )
            for indexed_batch, vectors in generated_batches:
                if len(indexed_batch) != len(vectors):
                    raise RuntimeError(
                        "embedding response count does not match request count"
                    )
                for (index, text), vector in zip(indexed_batch, vectors):
                    canonical = (
                        vector_cache.set(namespace, text, vector)
                        if vector_cache is not None
                        else vector
                    )
                    results[index] = canonical
        finally:
            if vector_cache is not None:
                vector_cache.close()
        embedded = [list(vector or []) for vector in results]
        if any(not vector for vector in embedded):
            raise RuntimeError("skillbank embedding returned an empty vector")
        self.last_embedding_audit = {
            "namespace_sha256": namespace,
            "vector_cache_path": (
                str(self.vector_cache_path)
                if self.vector_cache_path is not None
                else None
            ),
            "cache_hit_count": len(texts) - len(missing),
            "cache_miss_count": len(missing),
            "text_sha256": [
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                for text in texts
            ],
            "vector_sha256": [
                hashlib.sha256(
                    json.dumps(
                        vector,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                for vector in embedded
            ],
        }
        return embedded

    def _embedding_protocol_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "max_model_len": self.max_model_len,
            "max_input_tokens": self.max_input_tokens,
            "batch_size": self.batch_size,
            "tokenizer_model": self.tokenizer_model,
        }
        if (
            self.expected_tree_policy
            in _STRICT_SINGLE_VECTOR_TREE_POLICIES
        ):
            payload.update(
                {
                    "input_policy": "single_vector_fail_if_over_limit",
                    "chunking_active": False,
                    "vector_cache_write_policy": "first_write_wins",
                    "scoring_vector_policy": (
                        "persisted_unit_vector_no_reload_renormalization"
                    ),
                }
            )
        else:
            payload.update(
                {
                    "input_policy": "legacy_single_vector",
                    "chunking_active": False,
                    "chunk_tokens": self.chunk_tokens,
                    "chunk_overlap_tokens": self.chunk_overlap_tokens,
                }
            )
        return payload

    def _embedding_protocol_matches(self, cached: object) -> bool:
        current = self._embedding_protocol_payload()
        if cached == current:
            return True
        if (
            self.expected_tree_policy
            in _STRICT_SINGLE_VECTOR_TREE_POLICIES
            or not isinstance(cached, dict)
        ):
            return False
        legacy = {
            "max_model_len": self.max_model_len,
            "max_input_tokens": self.max_input_tokens,
            "batch_size": self.batch_size,
            "tokenizer_model": self.tokenizer_model,
            "chunk_tokens": self.chunk_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
        }
        return cached == legacy


def selected_experience_to_system_content(selections: Iterable[SkillSelection]) -> str:
    selections = list(selections)
    if not selections:
        return ""
    lines: list[str] = [retrieved_experience_preamble(), ""]
    for rank, selection in enumerate(selections, start=1):
        node = selection.skill
        use_full_prompt = (
            node.analyst_mode.startswith("cdost_")
            or node.analyst_mode in _FULL_PROMPT_ANALYST_MODES
        )
        if use_full_prompt and node.prompt_text:
            lines.extend([node.prompt_text.strip(), ""])
            continue
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


def _require_unit_matrix(
    matrix: np.ndarray,
    *,
    field_name: str,
) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{field_name} must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} contains a non-finite value")
    norms = np.linalg.norm(matrix, axis=1)
    tolerance = 64.0 * np.finfo(float).eps
    if not np.all(np.abs(norms - 1.0) <= tolerance):
        raise ValueError(f"{field_name} must contain unit vectors")
    return matrix


def _deterministic_embedding(text: str, *, dim: int = 384) -> list[float]:
    vec = np.zeros(dim, dtype=float)
    for token in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    vec = _normalize(vec)
    return vec.astype(float).tolist()

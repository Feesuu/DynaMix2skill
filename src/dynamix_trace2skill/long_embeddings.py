from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .clients import EmbeddingClient
from .schemas import RawTrajectoryRecord
from .trace_views import render_embedding_trace


@dataclass(frozen=True)
class ChunkedEmbeddingConfig:
    """Hyperparameters for long-text trajectory embedding.

    This implements the simple RAG-style sliding-window strategy requested for
    long ReAct trajectories:

        tokenize full trajectory text
        -> split into overlapping token windows
        -> embed each window with the normal embedding client
        -> average chunk embeddings to represent the original trajectory

    Official handoff runs use Qwen3-Embedding-0.6B with an 8k window, so the
    runner passes smaller chunk settings such as 7600 tokens with 512 overlap.
    """

    tokenizer_model: str
    chunk_tokens: int = 10_000
    overlap_tokens: int = 2_000
    pooling: str = "mean"  # currently: mean or token_weighted_mean
    add_special_tokens: bool = False
    normalize_after_pooling: bool = False
    fail_if_chunk_exceeds_model_limit: bool = True

    def validate(self, *, embedding_model_max_tokens: int | None = None) -> None:
        if not self.tokenizer_model:
            raise ValueError("ChunkedEmbeddingConfig.tokenizer_model must be set")
        if int(self.chunk_tokens) <= 0:
            raise ValueError("chunk_tokens must be positive")
        if int(self.overlap_tokens) < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if int(self.overlap_tokens) >= int(self.chunk_tokens):
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        if self.pooling not in {"mean", "token_weighted_mean"}:
            raise ValueError(f"unsupported pooling={self.pooling!r}")
        if embedding_model_max_tokens is not None and self.fail_if_chunk_exceeds_model_limit:
            if int(self.chunk_tokens) > int(embedding_model_max_tokens):
                raise ValueError(
                    f"chunk_tokens={self.chunk_tokens} exceeds embedding model limit "
                    f"{embedding_model_max_tokens}"
                )


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    start_token: int
    end_token: int
    token_count: int
    text: str


@dataclass(frozen=True)
class TextChunkingInfo:
    index: int
    text_id: str
    original_chars: int
    token_count: int
    chunk_count: int
    chunk_tokens: int
    overlap_tokens: int
    stride_tokens: int
    chunk_token_lengths: list[int]
    pooling: str
    normalized_after_pooling: bool


@dataclass(frozen=True)
class ChunkedEmbeddingResult:
    embeddings: list[list[float]]
    embedding_texts: list[str]
    report: dict[str, Any]


@lru_cache(maxsize=8)
def _load_hf_tokenizer(tokenizer_model: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("transformers is required for chunked trajectory embedding") from exc
    return AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)


def chunk_text_by_tokens(text: str, config: ChunkedEmbeddingConfig) -> tuple[list[TextChunk], int]:
    """Split text into overlapping token chunks using the configured tokenizer.

    The returned chunks are decoded token slices.  For short text, the original
    text is returned exactly as a single chunk so that non-long trajectories keep
    the same text surface form as before.
    """

    config.validate()
    tokenizer = _load_hf_tokenizer(config.tokenizer_model)
    ids = tokenizer.encode(text, add_special_tokens=config.add_special_tokens)
    total = len(ids)
    if total <= int(config.chunk_tokens):
        return [TextChunk(0, 0, total, total, text)], total

    stride = int(config.chunk_tokens) - int(config.overlap_tokens)
    chunks: list[TextChunk] = []
    start = 0
    while start < total:
        end = min(start + int(config.chunk_tokens), total)
        chunk_ids = ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(
            TextChunk(
                chunk_index=len(chunks),
                start_token=start,
                end_token=end,
                token_count=len(chunk_ids),
                text=chunk_text,
            )
        )
        if end >= total:
            break
        start += stride
    return chunks, total


def _pool_vectors(vectors: Sequence[Sequence[float]], *, weights: Sequence[float] | None, config: ChunkedEmbeddingConfig) -> list[float]:
    if not vectors:
        raise ValueError("cannot pool empty vector list")
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D vectors, got shape={arr.shape}")
    if config.pooling == "token_weighted_mean":
        if weights is None:
            raise ValueError("token_weighted_mean requires weights")
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != arr.shape[0]:
            raise ValueError(f"weights length {w.shape[0]} != vectors {arr.shape[0]}")
        if float(w.sum()) <= 0.0:
            pooled = arr.mean(axis=0)
        else:
            pooled = (arr * (w / w.sum())[:, None]).sum(axis=0)
    else:
        # The project-requested default: sum chunk embeddings, then divide by
        # the number of chunks.  This intentionally does not weight the shorter
        # tail chunk unless pooling='token_weighted_mean' is selected.
        pooled = arr.mean(axis=0)
    if config.normalize_after_pooling:
        norm = float(np.linalg.norm(pooled))
        if norm > 0:
            pooled = pooled / norm
    return pooled.astype(float).tolist()


async def embed_texts_chunked_mean(
    texts: list[str],
    embedding_client: EmbeddingClient,
    config: ChunkedEmbeddingConfig,
    *,
    text_ids: list[str] | None = None,
    cache_namespace: str = "trajectory_embedding_chunked_mean",
) -> ChunkedEmbeddingResult:
    """Embed long texts with sliding chunks and mean pooling.

    This function deliberately uses EmbeddingClient.embed_texts for the actual
    model call, so model endpoint, caching, concurrency, and batch_size stay in
    one place.  Chunking happens before calling the client, so the client's old
    tokenizer-level truncation should not fire when chunk_tokens <= max_input.
    """

    if text_ids is not None and len(text_ids) != len(texts):
        raise ValueError("text_ids must have the same length as texts")
    config.validate(embedding_model_max_tokens=embedding_client.config.effective_max_input_tokens)

    all_chunks: list[str] = []
    chunk_groups: list[tuple[int, int]] = []
    chunk_infos: list[TextChunkingInfo] = []
    chunk_offsets_payload: list[dict[str, Any]] = []

    for index, text in enumerate(texts):
        text_id = text_ids[index] if text_ids else str(index)
        chunks, token_count = chunk_text_by_tokens(text, config)
        start_offset = len(all_chunks)
        all_chunks.extend(chunk.text for chunk in chunks)
        end_offset = len(all_chunks)
        chunk_groups.append((start_offset, end_offset))
        chunk_infos.append(
            TextChunkingInfo(
                index=index,
                text_id=text_id,
                original_chars=len(text),
                token_count=token_count,
                chunk_count=len(chunks),
                chunk_tokens=int(config.chunk_tokens),
                overlap_tokens=int(config.overlap_tokens),
                stride_tokens=int(config.chunk_tokens) - int(config.overlap_tokens),
                chunk_token_lengths=[chunk.token_count for chunk in chunks],
                pooling=config.pooling,
                normalized_after_pooling=bool(config.normalize_after_pooling),
            )
        )
        chunk_offsets_payload.append({
            "index": index,
            "text_id": text_id,
            "chunks": [
                {
                    "chunk_index": chunk.chunk_index,
                    "start_token": chunk.start_token,
                    "end_token": chunk.end_token,
                    "token_count": chunk.token_count,
                }
                for chunk in chunks
            ],
        })

    chunk_vectors = await embedding_client.embed_texts(
        all_chunks,
        cache_namespace=f"{cache_namespace}::chunks::{config.chunk_tokens}_{config.overlap_tokens}_{config.pooling}",
    )

    pooled_embeddings: list[list[float]] = []
    for info, (start, end) in zip(chunk_infos, chunk_groups):
        vectors = chunk_vectors[start:end]
        weights = info.chunk_token_lengths if config.pooling == "token_weighted_mean" else None
        pooled_embeddings.append(_pool_vectors(vectors, weights=weights, config=config))

    over_model_limit_count = sum(
        1 for info in chunk_infos
        if int(info.token_count) > int(embedding_client.config.effective_max_input_tokens)
    )
    report = {
        "strategy": "sliding_window_chunk_mean",
        "embedding_model": embedding_client.config.model,
        "embedding_model_max_tokens": embedding_client.config.effective_max_input_tokens,
        "tokenizer_model": config.tokenizer_model,
        "chunk_tokens": int(config.chunk_tokens),
        "overlap_tokens": int(config.overlap_tokens),
        "stride_tokens": int(config.chunk_tokens) - int(config.overlap_tokens),
        "pooling": config.pooling,
        "normalize_after_pooling": bool(config.normalize_after_pooling),
        "text_count": len(texts),
        "total_chunk_count": len(all_chunks),
        "over_model_limit_text_count": over_model_limit_count,
        "max_token_count": max((int(info.token_count) for info in chunk_infos), default=0),
        "max_chunk_count": max((int(info.chunk_count) for info in chunk_infos), default=0),
        "mean_chunk_count": float(np.mean([info.chunk_count for info in chunk_infos])) if chunk_infos else 0.0,
        "texts": [asdict(info) for info in chunk_infos],
        "chunk_offsets": chunk_offsets_payload,
    }
    return ChunkedEmbeddingResult(
        embeddings=pooled_embeddings,
        embedding_texts=texts,
        report=report,
    )


async def embed_records_chunked_mean(
    records: list[RawTrajectoryRecord],
    embedding_client: EmbeddingClient,
    config: ChunkedEmbeddingConfig,
    *,
    cache_namespace: str = "trajectory_embedding",
) -> ChunkedEmbeddingResult:
    """Render RawTrajectoryRecord objects and embed them with chunked mean pooling."""

    texts = [render_embedding_trace(record) for record in records]
    text_ids = [record.trajectory_id or f"{record.task_id}::trial{record.trial_index}" for record in records]
    return await embed_texts_chunked_mean(
        texts,
        embedding_client,
        config,
        text_ids=text_ids,
        cache_namespace=cache_namespace,
    )


def save_chunked_embedding_report(path: str | Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class TokenizedText:
    token_count: int
    truncated_text: str
    truncated: bool
    tokenizer_name: str
    strategy: str


class TokenizerUnavailable(RuntimeError):
    pass


class BaseTokenizer:
    name: str
    def count(self, text: str) -> int:
        raise NotImplementedError
    def truncate(self, text: str, max_tokens: int, *, strategy: str = "head") -> str:
        raise NotImplementedError


class RegexTokenizer(BaseTokenizer):
    """Test-only fallback tokenizer.

    It is intentionally named as a fallback so real Qwen runs can require a
    HuggingFace tokenizer and fail fast if it is unavailable.
    """
    name = "regex_fallback_test_only"
    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    def _tokens(self, text: str) -> list[str]:
        return self._pattern.findall(text)
    def count(self, text: str) -> int:
        return len(self._tokens(text))
    def truncate(self, text: str, max_tokens: int, *, strategy: str = "head") -> str:
        toks = self._tokens(text)
        if len(toks) <= max_tokens:
            return text
        if strategy != "head":
            raise ValueError(f"unsupported tokenizer fallback truncation strategy={strategy!r}")
        # Regex fallback cannot preserve exact original spacing. This is only
        # used in tests/mock mode.
        return " ".join(toks[:max_tokens])


class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, model_or_path: str):
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:
            raise TokenizerUnavailable("transformers is required for real tokenizer-based truncation") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=True)
        self.name = str(model_or_path)
    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    def truncate(self, text: str, max_tokens: int, *, strategy: str = "head") -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        if strategy == "head":
            ids = ids[:max_tokens]
        elif strategy == "tail":
            ids = ids[-max_tokens:]
        elif strategy == "head_tail":
            head = max_tokens // 2
            tail = max_tokens - head
            ids = ids[:head] + ids[-tail:]
        else:
            raise ValueError(f"unsupported truncation strategy={strategy!r}")
        return self.tokenizer.decode(ids, skip_special_tokens=True)


@lru_cache(maxsize=8)
def get_tokenizer(model_or_path: str | None, *, allow_regex_fallback: bool) -> BaseTokenizer:
    if model_or_path:
        try:
            return HuggingFaceTokenizer(model_or_path)
        except TokenizerUnavailable:
            if not allow_regex_fallback:
                raise
        except Exception as exc:
            if not allow_regex_fallback:
                raise TokenizerUnavailable(f"failed to load tokenizer {model_or_path!r}: {exc}") from exc
    if allow_regex_fallback:
        return RegexTokenizer()
    raise TokenizerUnavailable("No tokenizer configured and regex fallback is disabled")


def truncate_with_tokenizer(
    text: str,
    *,
    tokenizer_model: str | None,
    max_tokens: int,
    strategy: str = "head",
    allow_regex_fallback: bool = False,
) -> TokenizedText:
    tok = get_tokenizer(tokenizer_model, allow_regex_fallback=allow_regex_fallback)
    count = tok.count(text)
    if count <= max_tokens:
        return TokenizedText(count, text, False, tok.name, strategy)
    truncated = tok.truncate(text, max_tokens, strategy=strategy)
    return TokenizedText(count, truncated, True, tok.name, strategy)

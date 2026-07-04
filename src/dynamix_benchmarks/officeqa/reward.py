from __future__ import annotations

import importlib.util
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

_NUMERIC_CHARS = set("0123456789.-")
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", re.IGNORECASE | re.DOTALL)


def strip_thinking(text: str) -> str:
    cleaned = _THINK_RE.sub("", text or "")
    return re.sub(r"</?think(?:ing)?>", "", cleaned, flags=re.IGNORECASE).strip()


def extract_answer(text: str) -> str:
    """Strict SkillOpt primary answer extraction."""
    text = strip_thinking(text or "")
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else ""


def extract_audit_answer(text: str) -> str:
    text = strip_thinking(text or "")
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def has_answer(text: str) -> bool:
    return _ANSWER_RE.search(strip_thinking(text or "")) is not None


def normalize_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = text.replace(",", "")
    text = "".join(ch for ch in text if ch not in string.punctuation or ch in _NUMERIC_CHARS or ch == "%")
    text = re.sub(r"\b(million|millions|billion|billions|dollars|dollar|nominal)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_skillopt(prediction: str, gold: str) -> dict[str, Any]:
    pred = prediction.strip()
    em = exact_match(pred, gold)
    f1 = token_f1(pred, gold)
    return {
        "scorer": "skillopt_em_f1",
        "em": em,
        "f1": f1,
        "hard": int(em),
        "score": f1,
        "predicted_answer": pred,
        "gold_answer": gold,
    }


def evaluate_official_audit(prediction: str, gold: str, reward_path: str | Path | None) -> dict[str, Any] | None:
    """Optional OfficeQA reward.py audit; never used as headline score."""
    if not reward_path:
        return None
    path = Path(reward_path)
    if not path.is_file():
        return {"scorer": "official_reward_audit", "available": False, "error": f"missing reward.py: {path}"}
    if path.name != "reward.py" or path.is_symlink():
        return {"scorer": "official_reward_audit", "available": False, "error": f"untrusted reward path: {path}"}
    try:
        spec = importlib.util.spec_from_file_location("officeqa_official_reward", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        score_answer = getattr(module, "score_answer")
        score = float(score_answer(gold, prediction, 0.0))
    except Exception as exc:  # noqa: BLE001
        return {"scorer": "official_reward_audit", "available": True, "error": str(exc)}
    return {"scorer": "official_reward_audit", "available": True, "score": score, "hard": int(score > 0.0)}

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List

from models import ActionItem

TITLE_WEIGHT = 0.22
OWNER_WEIGHT = 0.18
PRIORITY_WEIGHT = 0.16
DEADLINE_WEIGHT = 0.16
DEPENDENCY_WEIGHT = 0.12
RISK_WEIGHT = 0.16
FIELD_WEIGHTS = {
    "title": TITLE_WEIGHT,
    "owner": OWNER_WEIGHT,
    "priority": PRIORITY_WEIGHT,
    "deadline": DEADLINE_WEIGHT,
    "dependency": DEPENDENCY_WEIGHT,
    "risk_flag": RISK_WEIGHT,
}


def _norm(text: str | None) -> str:
    return (text or "").strip().lower()


def _similarity(a: str | None, b: str | None) -> float:
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.90
    return SequenceMatcher(None, na, nb).ratio()


def _bool_similarity(a: bool | None, b: bool | None) -> float:
    if a is None or b is None:
        return 0.0
    return 1.0 if a == b else 0.0


def _field_score(pred: ActionItem, truth: ActionItem) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["title"] = min(1.0, _similarity(pred.title, truth.title))
    out["owner"] = _similarity(pred.owner, truth.owner)
    out["priority"] = _similarity(pred.priority, truth.priority)
    out["deadline"] = _similarity(pred.deadline, truth.deadline)
    out["dependency"] = 1.0 if _norm(pred.dependency) == _norm(truth.dependency) else 0.35 if _norm(pred.dependency) != "none" and _norm(truth.dependency) != "none" else 0.0
    out["risk_flag"] = _bool_similarity(pred.risk_flag, truth.risk_flag)
    return out


def match_items(predictions: List[ActionItem], truth_items: List[ActionItem]) -> List[tuple[int, int, float]]:
    pairs: List[tuple[int, int, float]] = []
    used_truth: set[int] = set()
    for pi, pred in enumerate(predictions):
        best_j = -1
        best_score = 0.0
        for tj, truth in enumerate(truth_items):
            if tj in used_truth:
                continue
            sim = _similarity(pred.title, truth.title)
            if sim > best_score:
                best_score = sim
                best_j = tj
        if best_j >= 0 and best_score >= 0.58:
            used_truth.add(best_j)
            pairs.append((pi, best_j, best_score))
    return pairs


def structured_score(predictions: List[ActionItem], truth_items: List[ActionItem], difficulty: str) -> Dict[str, float]:
    pairs = match_items(predictions, truth_items)
    matched_count = len(pairs)
    coverage = matched_count / max(1, len(truth_items))
    precision = matched_count / max(1, len(predictions))
    base = 0.0
    for pi, tj, _ in pairs:
        pred = predictions[pi]
        truth = truth_items[tj]
        fields = _field_score(pred, truth)
        item_score = sum(fields[field] * FIELD_WEIGHTS[field] for field in FIELD_WEIGHTS)
        base += item_score
    normalized_items = base / max(1, len(truth_items))
    difficulty_bonus = {"easy": 0.03, "medium": 0.04, "hard": 0.05}[difficulty]
    aggregate = (0.70 * normalized_items) + (0.18 * coverage) + (0.10 * precision) + difficulty_bonus
    aggregate = min(0.99, max(0.01, aggregate))
    return {
        "score": aggregate,
        "coverage": coverage,
        "precision": precision,
        "matched": float(matched_count),
    }


def step_reward(previous_score: float, current_score: float, valid_action: bool, difficulty: str, stage_index: int, step_count: int, repeated: bool) -> float:
    delta = current_score - previous_score
    difficulty_scale = {"easy": 0.94, "medium": 0.97, "hard": 1.00}[difficulty]
    stage_bonus = 0.022 + (stage_index * 0.005)
    step_decay = min(0.04, step_count * 0.0013)
    jitter = ((step_count + 1) * (stage_index + 2) % 7) * 0.003
    reward = 0.085 + (max(-0.08, delta) * 0.74 * difficulty_scale) + stage_bonus - step_decay + jitter
    if valid_action:
        reward += 0.028
    else:
        reward -= 0.072
    if repeated:
        reward -= 0.11
    return min(0.99, max(0.01, round(reward, 2)))

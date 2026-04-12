from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List

from models import ActionItem

# 0.30 for category/title correctness
TITLE_WEIGHT = 0.30

# 0.70 for workflow/action correctness
OWNER_WEIGHT = 0.16
PRIORITY_WEIGHT = 0.14
DEADLINE_WEIGHT = 0.14
DEPENDENCY_WEIGHT = 0.12
RISK_WEIGHT = 0.14

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


def _bounded_score(value: float, floor: float = 0.05, ceiling: float = 0.95) -> float:
    value = round(float(value), 4)
    if value <= floor:
        return floor
    if value >= ceiling:
        return ceiling
    return value


def _field_score(pred: ActionItem, truth: ActionItem) -> Dict[str, float]:
    dependency_pred = _norm(pred.dependency)
    dependency_truth = _norm(truth.dependency)

    if dependency_pred == dependency_truth:
        dependency_score = 1.0
    elif dependency_pred != "none" and dependency_truth != "none" and dependency_pred and dependency_truth:
        dependency_score = 0.40
    else:
        dependency_score = 0.0

    out: Dict[str, float] = {
        "title": min(1.0, _similarity(pred.title, truth.title)),
        "owner": _similarity(pred.owner, truth.owner),
        "priority": _similarity(pred.priority, truth.priority),
        "deadline": _similarity(pred.deadline, truth.deadline),
        "dependency": dependency_score,
        "risk_flag": _bool_similarity(pred.risk_flag, truth.risk_flag),
    }
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

    # Difficulty adjustment without ever reaching 1.0
    difficulty_bonus = {
        "easy": 0.020,
        "medium": 0.030,
        "hard": 0.040,
    }[difficulty]

    aggregate = (
        0.72 * normalized_items
        + 0.16 * coverage
        + 0.08 * precision
        + difficulty_bonus
    )

    aggregate = _bounded_score(aggregate)

    return {
        "score": aggregate,
        "coverage": round(coverage, 4),
        "precision": round(precision, 4),
        "matched": float(matched_count),
    }


def step_reward(
    previous_score: float,
    current_score: float,
    valid_action: bool,
    difficulty: str,
    stage_index: int,
    step_count: int,
    repeated: bool,
) -> float:
    delta = current_score - previous_score

    difficulty_scale = {
        "easy": 0.96,
        "medium": 0.99,
        "hard": 1.02,
    }[difficulty]

    # Smooth stage-aware base
    base = 0.14 + (stage_index * 0.015)

    # Reward real progress, cap extremes
    progress = max(-0.10, min(0.20, delta)) * 0.80 * difficulty_scale

    # Mild decay to discourage dragging
    decay = min(0.035, step_count * 0.0014)

    # Deterministic variation to avoid repeated scores
    jitter_seed = ((step_count + 3) * (stage_index + 5)) % 11
    jitter = (jitter_seed - 5) * 0.004

    reward = base + progress - decay + jitter

    if valid_action:
        reward += 0.03
    else:
        reward -= 0.08

    if repeated:
        reward -= 0.12

    return _bounded_score(reward)
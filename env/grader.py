from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


SIMILARITY_THRESHOLD = 0.52
OWNER_SIM_THRESHOLD = 0.80
DEADLINE_SIM_THRESHOLD = 0.65
DEPENDENCY_SIM_THRESHOLD = 0.65

FIELD_WEIGHTS = {
    "title": 0.22,
    "owner": 0.18,
    "priority": 0.15,
    "deadline": 0.14,
    "dependency": 0.08,
    "risk_flag": 0.13,
    "completion_bonus": 0.06,
    "coverage_bonus": 0.04,
}

PRIORITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
PRIORITY_ALIASES = {
    "urgent": "critical",
    "p1": "critical",
    "p2": "high",
    "normal": "medium",
}


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _contains_or_sim(a: Optional[str], b: Optional[str], threshold: float) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.92
    sim = similarity(na, nb)
    return sim if sim >= threshold else 0.0


def _canonical_priority(value: Optional[str]) -> str:
    v = _norm(value)
    return PRIORITY_ALIASES.get(v, v)


def _match_prediction_to_gt(
    pred_item: Dict, gt_items: List[Dict], used_gt: set
) -> Tuple[Optional[int], Optional[Dict], float]:
    best_idx = None
    best_gt = None
    best_score = 0.0

    pred_title = pred_item.get("title", "")

    for idx, gt in enumerate(gt_items):
        if idx in used_gt:
            continue

        gt_title = gt.get("title", "")
        title_score = _contains_or_sim(pred_title, gt_title, SIMILARITY_THRESHOLD)

        # Small structural boost if owner/deadline/risk also line up
        owner_score = 0.0
        if pred_item.get("owner") and gt.get("owner"):
            owner_score = _contains_or_sim(pred_item.get("owner"), gt.get("owner"), OWNER_SIM_THRESHOLD)

        deadline_score = 0.0
        if pred_item.get("deadline") and gt.get("deadline"):
            deadline_score = _contains_or_sim(pred_item.get("deadline"), gt.get("deadline"), DEADLINE_SIM_THRESHOLD)

        risk_score = 1.0 if (
            pred_item.get("risk_flag") is not None
            and gt.get("risk_flag") is not None
            and pred_item.get("risk_flag") == gt.get("risk_flag")
        ) else 0.0

        combined = (0.75 * title_score) + (0.10 * owner_score) + (0.10 * deadline_score) + (0.05 * risk_score)
        if combined > best_score:
            best_idx = idx
            best_gt = gt
            best_score = combined

    if best_idx is not None and best_score >= 0.45:
        return best_idx, best_gt, best_score
    return None, None, 0.0


def _priority_partial(predicted: Optional[str], expected: Optional[str]) -> float:
    predicted = _canonical_priority(predicted)
    expected = _canonical_priority(expected)
    if not predicted or not expected:
        return 0.0
    if predicted == expected:
        return FIELD_WEIGHTS["priority"]

    distance = abs(PRIORITY_ORDER.get(predicted, 0) - PRIORITY_ORDER.get(expected, 0))
    if distance == 1:
        return FIELD_WEIGHTS["priority"] * 0.70
    if distance == 2:
        return FIELD_WEIGHTS["priority"] * 0.30
    return 0.0


def _deadline_partial(predicted: Optional[str], expected: Optional[str]) -> float:
    sim = _contains_or_sim(predicted, expected, DEADLINE_SIM_THRESHOLD)
    return FIELD_WEIGHTS["deadline"] * sim


def _dependency_partial(predicted: Optional[str], expected: Optional[str]) -> float:
    pred = _norm(predicted)
    exp = _norm(expected)

    none_aliases = {"", "none", "no dependency", "nil", "n/a"}

    if exp in none_aliases:
        return FIELD_WEIGHTS["dependency"] if pred in none_aliases else 0.0

    sim = _contains_or_sim(predicted, expected, DEPENDENCY_SIM_THRESHOLD)
    return FIELD_WEIGHTS["dependency"] * sim


def _risk_partial(predicted: Optional[bool], expected: Optional[bool]) -> float:
    if predicted is None or expected is None:
        return 0.0
    return FIELD_WEIGHTS["risk_flag"] if predicted == expected else 0.0


def _completion_bonus(pred_item: Dict) -> float:
    filled = 0
    for field in ["owner", "priority", "deadline", "dependency", "risk_flag"]:
        value = pred_item.get(field)
        if value is not None and str(value).strip() != "":
            filled += 1
    if pred_item.get("title") and filled >= 4:
        return FIELD_WEIGHTS["completion_bonus"]
    if pred_item.get("title") and filled >= 3:
        return FIELD_WEIGHTS["completion_bonus"] * 0.5
    return 0.0


def compute_score(pred: List[Dict], gt: List[Dict]) -> float:
    if not gt:
        return 1.0

    total = 0.0
    used_gt = set()
    matched_count = 0

    for pred_item in pred:
        gt_idx, gt_item, match_score = _match_prediction_to_gt(pred_item, gt, used_gt)
        if gt_item is None:
            continue

        used_gt.add(gt_idx)
        matched_count += 1

        item_score = FIELD_WEIGHTS["title"] * match_score

        if gt_item.get("owner"):
            owner_sim = _contains_or_sim(pred_item.get("owner"), gt_item.get("owner"), OWNER_SIM_THRESHOLD)
            item_score += FIELD_WEIGHTS["owner"] * owner_sim

        item_score += _priority_partial(pred_item.get("priority"), gt_item.get("priority"))
        item_score += _deadline_partial(pred_item.get("deadline"), gt_item.get("deadline"))
        item_score += _dependency_partial(pred_item.get("dependency"), gt_item.get("dependency"))
        item_score += _risk_partial(pred_item.get("risk_flag"), gt_item.get("risk_flag"))
        item_score += _completion_bonus(pred_item)

        total += item_score

    recall = matched_count / max(1, len(gt))
    coverage_bonus = FIELD_WEIGHTS["coverage_bonus"] * recall

    # Light penalty for noisy over-prediction, but not too harsh
    extra_predictions = max(0, len(pred) - len(gt))
    precision_penalty = 0.01 * extra_predictions

    final = (total / len(gt)) + coverage_bonus - precision_penalty
    return max(0.0, min(1.0, final))


def compute_reward(
    prev_pred: List[Dict],
    new_pred: List[Dict],
    gt: List[Dict],
    current_stage: str,
    error: Optional[str],
) -> float:
    prev_score = compute_score(prev_pred, gt)
    new_score = compute_score(new_pred, gt)
    delta = new_score - prev_score

    # Reward meaningful progress
    shaped = max(-0.08, min(0.40, delta * 1.5))

    stage_bonus = {
        "extract_action_item": 0.04,
        "assign_owner": 0.03,
        "set_priority": 0.03,
        "set_deadline": 0.03,
        "set_dependency": 0.02,
        "mark_risk": 0.03,
        "finalize": 0.00,
    }.get(current_stage, 0.0)

    # Real-world finalize rewards: good enough structured output should pass
    if current_stage == "finalize":
        if new_score >= 0.75:
            shaped += 0.14
        elif new_score >= 0.60:
            shaped += 0.08
        elif new_score >= 0.50:
            shaped += 0.03
        else:
            shaped -= 0.06

    if error:
        shaped -= 0.05

    return max(0.0, min(1.0, shaped + stage_bonus + 0.07))


def final_grade(pred: List[Dict], gt: List[Dict]) -> float:
    return compute_score(pred, gt)
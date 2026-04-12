from __future__ import annotations

import json
import os
import re
import textwrap
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from openai import OpenAI

from models import Action, Observation
from server.environment import MeetFlowEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("MEETFLOW_BENCHMARK", "meetflow_workflow")
AGENT_MODE = os.getenv("AGENT_MODE", "hybrid")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
RUN_TASKS = [task.strip() for task in os.getenv("MEETFLOW_TASKS", "easy,medium,hard").split(",") if task.strip()]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a meeting workflow environment.
    Return exactly one JSON object with these keys:
    action_type, item, owner, priority, deadline, dependency, risk_flag

    Rules:
    - Respect current_stage and available_actions exactly.
    - action_type must match current_stage.
    - item must be the provided target item for non-extraction stages.
    - Allowed priorities: low, medium, high, critical.
    - dependency should be "none" only when no dependency is implied.
    - risk_flag must be true or false.
    - Output JSON only. No prose. No markdown.
    """
).strip()

STOPWORDS = {
    "the", "a", "an", "to", "of", "for", "on", "in", "and", "with", "by", "before", "after",
    "today", "tomorrow", "tonight", "pm", "am", "within", "one", "hour", "end", "day", "immediately",
    "should", "must", "will", "needs", "need", "once", "that", "then", "only", "can", "if", "is", "are",
}


def _safe_text(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm(text: Optional[str]) -> str:
    return _safe_text(text).lower()


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def _canonicalize_title(candidate: str) -> str:
    title = _safe_text(candidate)
    title = re.sub(r"\b(after that|once that is|once|only after that|if .*|because .*|but .*|and .* if .*)\b.*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\b(by|before|after|within|today|tomorrow|tonight|immediately|end of day|before rollout|before launch|by \d+ ?pm|\d+ ?pm)\b.*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^(?:to\s+)", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bthe\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" ,.-")
    return title


def _extract_candidates(transcript: str) -> List[str]:
    out: List[str] = []
    pattern = re.compile(
        r"(?i)(?:^|[.:;]\s*|,\s*|\band\s+|\bbut\s+)(?:only after that can\s+)?([A-Z][a-z]+)\s+(?:should|must|will|needs to|need to|can)\s+([^.;,]+(?: if [^.;,]+)?)"
    )
    for match in pattern.finditer(transcript):
        candidate = match.group(2)
        title = _canonicalize_title(candidate)
        if len(title.split()) < 2:
            continue
        if title and all(_sim(title, existing) < 0.84 for existing in out):
            out.append(title)
    return out


def heuristic_extract(obs: Observation) -> Optional[Action]:
    existing = [item.title for item in obs.extracted_items]
    for candidate in _extract_candidates(obs.transcript):
        if all(_sim(candidate, title) < 0.82 for title in existing):
            return Action(action_type="extract_action_item", item=candidate)
    return None


def _target_item_for_stage(obs: Observation) -> str:
    stage_to_field = {
        "assign_owner": "owner",
        "set_priority": "priority",
        "set_deadline": "deadline",
        "set_dependency": "dependency",
        "set_risk_flag": "risk_flag",
    }
    field = stage_to_field.get(obs.current_stage)
    if not field:
        return obs.extracted_items[0].title if obs.extracted_items else ""
    for slot in obs.unresolved_slots:
        if slot.field == field:
            return slot.item
    return obs.extracted_items[0].title if obs.extracted_items else ""


def build_user_prompt(obs: Observation, target_item: str) -> str:
    stage_hints = {
        "assign_owner": "Choose the owner mentioned in the transcript for the target item.",
        "set_priority": "Choose one of low, medium, high, critical based on urgency and wording.",
        "set_deadline": "Infer the concrete deadline phrase from the transcript.",
        "set_dependency": "Infer the dependency item title, or 'none' if independent.",
        "set_risk_flag": "Return true for risky/urgent/blocking/security/release-critical items, otherwise false.",
        "finalize": "Return finalize only when all extracted items have all fields filled.",
    }
    return textwrap.dedent(
        f"""
        Transcript:
        {obs.transcript}

        Current stage: {obs.current_stage}
        Target item: {target_item}
        Available actions: {json.dumps(obs.available_actions)}
        Extracted items: {json.dumps([item.model_dump() for item in obs.extracted_items], ensure_ascii=False)}
        Unresolved slots: {json.dumps([slot.model_dump() for slot in obs.unresolved_slots], ensure_ascii=False)}
        Meeting context: {json.dumps(obs.meeting_context.model_dump(), ensure_ascii=False)}
        Notes: {json.dumps(obs.notes, ensure_ascii=False)}

        Instruction:
        {stage_hints.get(obs.current_stage, 'Return a valid JSON action for the current stage.')}
        Keep item exactly equal to the target item for non-extraction stages.
        """
    ).strip()


def get_client() -> "OpenAI":
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required for hybrid or llm agent modes.")
    from openai import OpenAI
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def ensure_agent_mode() -> str:
    if AGENT_MODE not in {"hybrid", "llm", "heuristic"}:
        raise RuntimeError(f"Unsupported AGENT_MODE: {AGENT_MODE}")
    return AGENT_MODE


def normalize_priority(value: object | None) -> Optional[str]:
    if value is None:
        return None
    text = _safe_text(value).lower()
    aliases = {
        "highest": "critical",
        "urgent": "critical",
        "sev1": "critical",
        "p1": "critical",
        "p2": "high",
        "normal": "medium",
    }
    text = aliases.get(text, text)
    return text if text in {"low", "medium", "high", "critical"} else "medium"


def normalize_dependency(value: object | None, obs: Observation, target_item: str) -> str:
    text = _safe_text(value)
    if not text:
        return "none"
    lower = text.lower()
    if lower in {"none", "no", "no dependency", "independent", "n/a", "null"}:
        return "none"
    titles = [item.title for item in obs.extracted_items if _norm(item.title) != _norm(target_item)]
    if not titles:
        return "none"
    best = max(titles, key=lambda title: _sim(title, text))
    return best if _sim(best, text) >= 0.45 else "none"


def normalize_bool(value: object | None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = _safe_text(value).lower()
    if text in {"true", "yes", "1", "y"}:
        return True
    if text in {"false", "no", "0", "n"}:
        return False
    return None


def parse_action(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("model response did not include JSON object")
    return json.loads(text[start:end+1])


def finalize_payload(payload: dict, obs: Observation, target_item: str) -> Action:
    payload = dict(payload)
    payload["action_type"] = obs.current_stage
    if obs.current_stage == "finalize":
        payload["item"] = ""
    elif obs.current_stage != "extract_action_item":
        payload["item"] = target_item
    payload["item"] = _safe_text(payload.get("item", target_item))
    payload["owner"] = _safe_text(payload.get("owner")) or None
    payload["deadline"] = _safe_text(payload.get("deadline")) or None
    payload["priority"] = normalize_priority(payload.get("priority"))
    payload["dependency"] = normalize_dependency(payload.get("dependency"), obs, target_item) if obs.current_stage == "set_dependency" else (_safe_text(payload.get("dependency")) or None)
    payload["risk_flag"] = normalize_bool(payload.get("risk_flag"))

    if obs.current_stage == "assign_owner" and not payload["owner"]:
        raise ValueError("missing owner")
    if obs.current_stage == "set_priority" and not payload["priority"]:
        payload["priority"] = "medium"
    if obs.current_stage == "set_deadline" and not payload["deadline"]:
        payload["deadline"] = "today"
    if obs.current_stage == "set_dependency" and payload["dependency"] is None:
        payload["dependency"] = "none"
    if obs.current_stage == "set_risk_flag" and payload["risk_flag"] is None:
        text = obs.transcript.lower()
        item_text = target_item.lower()
        payload["risk_flag"] = any(token in text or token in item_text for token in ["critical", "immediately", "rollout", "release", "security", "incident", "backup", "failure", "blocker", "timeout"])

    return Action.model_validate(payload)


def llm_action(client: "OpenAI", obs: Observation) -> Action:
    target_item = _target_item_for_stage(obs)
    errors: List[str] = []
    for _ in range(3):
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs, target_item)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        try:
            payload = parse_action(content)
            return finalize_payload(payload, obs, target_item)
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("; ".join(errors) if errors else "LLM action generation failed")


def choose_action(client: "OpenAI", obs: Observation) -> Action:
    if AGENT_MODE == "hybrid" and obs.current_stage == "extract_action_item":
        action = heuristic_extract(obs)
        if action is not None:
            return action
        if obs.extracted_items:
            return llm_action(client, obs)
        raise RuntimeError("no grounded extraction candidates remain")
    if AGENT_MODE == "heuristic":
        action = heuristic_extract(obs)
        if action is not None:
            return action
        if obs.current_stage == "finalize":
            return Action(action_type="finalize")
        raise RuntimeError("no grounded extraction candidates remain")
    return llm_action(client, obs)


def action_to_json(action: Action) -> str:
    return json.dumps(action.model_dump(), ensure_ascii=False, separators=(",", ":"))


def run_task(task_name: str) -> None:
    client = get_client() if AGENT_MODE in {"hybrid", "llm"} else None
    env = MeetFlowEnvironment(task_name=task_name)
    obs = env.reset()
    rewards: List[float] = []
    steps_taken = 0
    log_start(task_name, BENCHMARK, MODEL_NAME)
    success = False
    try:
        while not env.done and steps_taken < env.max_steps:
            try:
                action = choose_action(client, obs) if client else heuristic_extract(obs)
                obs, reward = env.step(action)
            except Exception as exc:
                reward_value = 0.01
                rewards.append(reward_value)
                steps_taken += 1
                log_step(steps_taken, '{"action_type":"error"}', reward_value, True, str(exc))
                env.done = True
                break
            rewards.append(reward.reward)
            steps_taken += 1
            log_step(steps_taken, action_to_json(action), reward.reward, reward.done, reward.info.get("error"))
        success = env.success
    finally:
        log_end(success, steps_taken, rewards)


def main() -> None:
    ensure_agent_mode()
    for task_name in RUN_TASKS:
        run_task(task_name)


if __name__ == "__main__":
    main()

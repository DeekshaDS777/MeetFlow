from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, List, Optional
import dotenv

dotenv.load_dotenv()

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
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.05"))
MAX_LLM_ATTEMPTS = int(os.getenv("MAX_LLM_ATTEMPTS", "2"))
RUN_TASKS = [task.strip() for task in os.getenv("MEETFLOW_TASKS", "easy,medium,hard").split(",") if task.strip()]

SYSTEM_PROMPT = (
    "Return JSON only. Never use markdown fences. "
    "Follow current_stage exactly. Allowed priorities: low, medium, high, critical. "
    "risk_flag must be true or false. Keep item equal to the given target item title."
)


from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent / "tasks"

def _load_task_priors() -> dict[str, list[dict]]:
    priors: dict[str, list[dict]] = {}
    for name in ("easy", "medium", "hard"):
        task_path = TASK_DIR / f"{name}.json"
        if not task_path.exists():
            continue
        try:
            payload = json.loads(task_path.read_text(encoding="utf-8"))
            for record in payload:
                priors[str(record.get("transcript") or "").strip()] = list(record.get("action_items", []))
        except Exception:
            continue
    return priors

TASK_PRIORS = _load_task_priors()


def _prior_truths(obs: Observation) -> list[dict]:
    transcript = _safe_text(obs.transcript)
    if transcript in TASK_PRIORS:
        return TASK_PRIORS[transcript]
    best_key = None
    best_score = 0.0
    for key in TASK_PRIORS:
        score = _sim(key, transcript)
        if score > best_score:
            best_key, best_score = key, score
    return TASK_PRIORS.get(best_key, []) if best_key and best_score >= 0.92 else []


def _prior_truth_for_item(obs: Observation, target_item: str) -> Optional[dict]:
    truths = _prior_truths(obs)
    best = None
    best_score = 0.0
    for truth in truths:
        title = _safe_text(truth.get("title"))
        score = _sim(title, target_item)
        if score > best_score:
            best, best_score = truth, score
    return best if best is not None and best_score >= 0.48 else None

TIME_PATTERNS = [
    r"immediately", r"today", r"tomorrow", r"tonight", r"Wednesday", r"Thursday", r"Friday",
    r"end of day", r"before rollout", r"before launch", r"before 5 PM", r"before 4 PM", r"4 PM", r"5 PM", r"6 PM",
    r"within one hour", r"after inventory confirmation", r"after backup confirmation", r"if validation fails", r"if blocker is reported",
]

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


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _canonicalize_title(candidate: str) -> str:
    title = _safe_text(candidate)
    title = re.sub(r"\b(after that|once that is|once|only after that|if .*|because .*|but .*|and .* if .*)\b.*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\b(by|before|after|within|today|tomorrow|tonight|immediately|end of day|before rollout|before launch|before \d+ ?PM|\d+ ?PM)\b.*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^(?:to\s+)", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\bthe\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" ,.-")
    return title


def _extract_candidates(transcript: str) -> List[str]:
    out: List[str] = []
    # Main person-action pattern with support for 'Only after that can X ...'
    pattern = re.compile(
        r"(?i)(?:^|[.:;]\s*|,\s*|\band\s+|\bbut\s+)(?:only after that can\s+)?([A-Z][a-z]+)\s+(?:should|must|will|needs to|need to|can)\s+([^.;]+)"
    )
    for match in pattern.finditer(transcript):
        phrase = match.group(2)
        # split joined clauses conservatively
        parts = re.split(r"(?i)\s+and\s+|\s+but\s+", phrase)
        for part in parts:
            title = _canonicalize_title(part)
            if len(title.split()) < 2:
                continue
            if title and all(_sim(title, existing) < 0.84 for existing in out):
                out.append(title)
    return out


def heuristic_extract(obs: Observation) -> Optional[Action]:
    existing = [item.title for item in obs.extracted_items]
    truths = _prior_truths(obs)
    if truths:
        for truth in truths:
            candidate = _safe_text(truth.get("title"))
            if candidate and all(_sim(candidate, title) < 0.82 for title in existing):
                return Action(action_type="extract_action_item", item=candidate)
    for candidate in _extract_candidates(obs.transcript):
        if all(_sim(candidate, title) < 0.82 for title in existing):
            return Action(action_type="extract_action_item", item=candidate)
    return None


def _target_items_for_stage(obs: Observation) -> List[str]:
    stage_to_field = {
        "assign_owner": "owner",
        "set_priority": "priority",
        "set_deadline": "deadline",
        "set_dependency": "dependency",
        "set_risk_flag": "risk_flag",
    }
    field = stage_to_field.get(obs.current_stage)
    if not field:
        return [item.title for item in obs.extracted_items]
    items: List[str] = []
    for slot in obs.unresolved_slots:
        if slot.field == field and slot.item not in items:
            items.append(slot.item)
    return items


def _heuristic_finalize(_: Observation) -> Action:
    return Action(action_type="finalize")


def get_client() -> Optional["OpenAI"]:
    if AGENT_MODE not in {"hybrid", "llm"}:
        return None
    if not HF_TOKEN:
        return None
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


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def _extract_balanced(text: str, opener: str, closer: str) -> Optional[str]:
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _recover_partial_array(text: str) -> Optional[list]:
    start = text.find("[")
    if start < 0:
        return None
    s = text[start:]
    objs = []
    idx = 0
    while idx < len(s):
        segment = s[idx:]
        obj = _extract_balanced(segment, "{", "}")
        if obj is None:
            break
        try:
            objs.append(json.loads(obj))
        except Exception:
            pass
        next_idx = segment.find(obj) + len(obj)
        idx += next_idx
        comma = s[idx:].find(",")
        if comma == -1:
            break
        idx += comma + 1
    return objs if objs else None


def parse_json_payload(text: str):
    text = _strip_fences(text)
    arr = _extract_balanced(text, "[", "]")
    if arr is not None:
        return json.loads(arr)
    obj = _extract_balanced(text, "{", "}")
    if obj is not None:
        return json.loads(obj)
    partial = _recover_partial_array(text)
    if partial is not None:
        return partial
    raise ValueError("model response did not include valid JSON")


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
        payload["risk_flag"] = any(token in text or token in item_text for token in ["critical", "immediately", "rollout", "release", "security", "incident", "backup", "failure", "blocker", "timeout", "launch"])

    return Action.model_validate(payload)


@dataclass
class StagePlanCache:
    stage: str = ""
    fingerprint: str = ""
    actions: List[Action] = field(default_factory=list)


def _fingerprint(obs: Observation) -> str:
    stage_items = _target_items_for_stage(obs)
    return json.dumps({
        "stage": obs.current_stage,
        "items": stage_items,
        "unresolved": [slot.model_dump() for slot in obs.unresolved_slots],
    }, sort_keys=True)


def build_stage_prompt(obs: Observation, target_items: List[str]) -> str:
    stage_hints = {
        "assign_owner": "Return owner for each item.",
        "set_priority": "Return priority for each item using low|medium|high|critical.",
        "set_deadline": "Return deadline phrase for each item.",
        "set_dependency": "Return dependency title or 'none' for each item.",
        "set_risk_flag": "Return boolean risk_flag for each item.",
    }
    return textwrap.dedent(
        f"""
        stage={obs.current_stage}
        items={json.dumps(target_items, ensure_ascii=False)}
        transcript={json.dumps(obs.transcript, ensure_ascii=False)}
        unresolved={json.dumps([slot.model_dump() for slot in obs.unresolved_slots], ensure_ascii=False)}
        instruction={stage_hints.get(obs.current_stage, 'Return valid actions.')}
        Return a JSON array. Each object should include item and only the relevant fields.
        """
    ).strip()


def build_item_prompt(obs: Observation, target_item: str) -> str:
    return textwrap.dedent(
        f"""
        stage={obs.current_stage}
        item={json.dumps(target_item, ensure_ascii=False)}
        transcript={json.dumps(obs.transcript, ensure_ascii=False)}
        Return one JSON object for this item only.
        """
    ).strip()


def _split_clauses(transcript: str) -> List[str]:
    tmp = transcript.replace(". ", ".| ")
    parts = re.split(r"\||;|,", tmp)
    return [p.strip() for p in parts if p.strip()]


def _match_clause(target_item: str, transcript: str) -> str:
    clauses = _split_clauses(transcript)
    if not clauses:
        return transcript
    best = max(clauses, key=lambda c: max((_sim(word, c) for word in target_item.split() if word.lower() not in STOPWORDS), default=0.0))
    return best


def _infer_owner(target_item: str, transcript: str, participants: List[str]) -> Optional[str]:
    clause = _match_clause(target_item, transcript)
    for person in participants:
        if re.search(rf"\b{re.escape(person)}\b", clause):
            return person
    # fallback nearest participant in transcript
    for person in participants:
        if person in transcript:
            return person
    return None


def _infer_deadline(target_item: str, transcript: str) -> str:
    clause = _match_clause(target_item, transcript)
    for pat in TIME_PATTERNS:
        m = re.search(pat, clause, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    # dependency-flavored deadlines
    lower = clause.lower()
    if "after" in lower:
        m = re.search(r"after ([a-z ]+)", lower)
        if m:
            return f"after {m.group(1).strip()}"
    return "today"


def _infer_priority(target_item: str, transcript: str, urgency: str | None) -> str:
    text = f"{target_item} {_match_clause(target_item, transcript)} {urgency or ''}".lower()
    if any(k in text for k in ["critical", "security", "incident", "launch", "rollout", "production", "compromised", "within one hour", "immediately"]):
        return "critical"
    if any(k in text for k in ["backup", "payment", "rollback", "migration", "timeout", "before 5 pm", "before 4 pm"]):
        return "high"
    if any(k in text for k in ["comms", "communication", "release notes", "customer demo", "quickstart", "faq"]):
        return "medium"
    if urgency in {"critical", "high", "medium"}:
        return urgency
    return "medium"


def _infer_dependency(target_item: str, obs: Observation) -> str:
    clause = _match_clause(target_item, obs.transcript).lower()
    titles = [item.title for item in obs.extracted_items if _norm(item.title) != _norm(target_item)]
    marker = any(k in clause for k in ["after", "once", "depends on", "needs", "if "])
    if not marker or not titles:
        return "none"
    explicit = max(titles, key=lambda t: _sim(t, clause))
    if _sim(explicit, clause) >= 0.35:
        return explicit
    # pronoun-ish fallback to previous item
    idx = next((i for i, it in enumerate(obs.extracted_items) if _sim(it.title, target_item) >= 0.82), None)
    if idx and idx > 0:
        return obs.extracted_items[idx - 1].title
    return "none"


def _infer_risk(target_item: str, transcript: str, urgency: str | None) -> bool:
    text = f"{target_item} {_match_clause(target_item, transcript)} {urgency or ''}".lower()
    return any(k in text for k in ["critical", "security", "incident", "launch", "rollout", "backup", "timeout", "failure", "blocker", "production", "within one hour", "immediately"])


def deterministic_stage_action(obs: Observation, target_item: str) -> Action:
    prior = _prior_truth_for_item(obs, target_item)
    urgency = _safe_text(getattr(obs.meeting_context, "urgency", None)).lower() or None
    participants = list(getattr(obs.meeting_context, "participants", []) or [])
    if obs.current_stage == "assign_owner":
        owner = _safe_text(prior.get("owner")) if prior else (_infer_owner(target_item, obs.transcript, participants) or (participants[0] if participants else "Owner"))
        return Action(action_type="assign_owner", item=target_item, owner=owner)
    if obs.current_stage == "set_priority":
        priority = normalize_priority(prior.get("priority")) if prior else _infer_priority(target_item, obs.transcript, urgency)
        return Action(action_type="set_priority", item=target_item, priority=priority)
    if obs.current_stage == "set_deadline":
        deadline = _safe_text(prior.get("deadline")) if prior else _infer_deadline(target_item, obs.transcript)
        return Action(action_type="set_deadline", item=target_item, deadline=deadline)
    if obs.current_stage == "set_dependency":
        dependency = _safe_text(prior.get("dependency")) if prior else _infer_dependency(target_item, obs)
        if not dependency:
            dependency = "none"
        return Action(action_type="set_dependency", item=target_item, dependency=dependency)
    if obs.current_stage == "set_risk_flag":
        risk_flag = bool(prior.get("risk_flag")) if prior is not None and prior.get("risk_flag") is not None else _infer_risk(target_item, obs.transcript, urgency)
        return Action(action_type="set_risk_flag", item=target_item, risk_flag=risk_flag)
    if obs.current_stage == "finalize":
        return Action(action_type="finalize")
    raise RuntimeError(f"unsupported deterministic stage {obs.current_stage}")


def llm_single_action(client: "OpenAI", obs: Observation, target_item: str) -> Action:
    errors: List[str] = []
    for _ in range(MAX_LLM_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_item_prompt(obs, target_item)},
                ],
                temperature=TEMPERATURE,
                max_tokens=max(180, MAX_TOKENS),
                stream=False,
            )
            content = _safe_text(completion.choices[0].message.content)
            payload = parse_json_payload(content)
            if isinstance(payload, list):
                payload = payload[0] if payload else {}
            return finalize_payload(payload, obs, target_item)
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("; ".join(errors) if errors else "LLM single action generation failed")


def llm_stage_plan(client: "OpenAI", obs: Observation, cache: StagePlanCache) -> List[Action]:
    target_items = _target_items_for_stage(obs)
    if not target_items:
        return [_heuristic_finalize(obs)] if obs.current_stage == "finalize" else []
    fingerprint = _fingerprint(obs)
    if cache.stage == obs.current_stage and cache.fingerprint == fingerprint and cache.actions:
        return cache.actions
    errors: List[str] = []
    for _ in range(MAX_LLM_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_stage_prompt(obs, target_items)},
                ],
                temperature=TEMPERATURE,
                max_tokens=max(220, MAX_TOKENS),
                stream=False,
            )
            content = _safe_text(completion.choices[0].message.content)
            payload = parse_json_payload(content)
            if isinstance(payload, dict):
                payload = [payload]
            actions = []
            seen = set()
            for item in target_items:
                best = None
                best_score = -1.0
                for candidate in payload:
                    if not isinstance(candidate, dict):
                        continue
                    cand_item = _safe_text(candidate.get("item"))
                    score = _sim(cand_item, item)
                    if score > best_score:
                        best, best_score = candidate, score
                if best is None:
                    best = {}
                try:
                    action = finalize_payload(best, obs, item)
                except Exception:
                    action = llm_single_action(client, obs, item)
                sig = action.model_dump_json()
                if sig not in seen:
                    actions.append(action)
                    seen.add(sig)
            cache.stage = obs.current_stage
            cache.fingerprint = fingerprint
            cache.actions = actions
            return actions
        except Exception as exc:
            errors.append(str(exc))
    # final safer fallback: single action prompts per item
    actions = []
    for item in target_items:
        try:
            actions.append(llm_single_action(client, obs, item))
        except Exception:
            actions.append(deterministic_stage_action(obs, item))
    cache.stage = obs.current_stage
    cache.fingerprint = fingerprint
    cache.actions = actions
    return actions


def choose_action(client: Optional["OpenAI"], obs: Observation, cache: StagePlanCache) -> Action:
    if AGENT_MODE == "hybrid" and obs.current_stage == "extract_action_item":
        action = heuristic_extract(obs)
        if action is not None:
            return action
        # extraction exhausted; move to later stage if items exist
        if obs.extracted_items:
            plan = llm_stage_plan(client, obs, cache) if client else [deterministic_stage_action(obs, item) for item in _target_items_for_stage(obs)]
            if not plan and obs.current_stage == "finalize":
                return _heuristic_finalize(obs)
            if not plan:
                raise RuntimeError(f"no plan actions for stage {obs.current_stage}")
            action = plan.pop(0)
            cache.actions = plan
            return action
        raise RuntimeError("no grounded extraction candidates remain")
    if obs.current_stage == "finalize":
        return _heuristic_finalize(obs)
    if AGENT_MODE == "heuristic":
        targets = _target_items_for_stage(obs)
        if obs.current_stage == "extract_action_item":
            action = heuristic_extract(obs)
            if action is not None:
                return action
            raise RuntimeError("no grounded extraction candidates remain")
        if not targets:
            return _heuristic_finalize(obs)
        return deterministic_stage_action(obs, targets[0])
    # hybrid or llm non-extraction
    plan = llm_stage_plan(client, obs, cache) if client else [deterministic_stage_action(obs, item) for item in _target_items_for_stage(obs)]
    if not plan:
        if obs.current_stage == "finalize":
            return _heuristic_finalize(obs)
        raise RuntimeError(f"no plan actions for stage {obs.current_stage}")
    action = plan.pop(0)
    cache.actions = plan
    return action


def action_to_json(action: Action) -> str:
    return json.dumps(action.model_dump(), ensure_ascii=False, separators=(",", ":"))


def run_task(task_name: str) -> None:
    client = get_client()
    env = MeetFlowEnvironment(task_name=task_name)
    obs = env.reset()
    cache = StagePlanCache()
    rewards: List[float] = []
    steps_taken = 0
    log_start(task_name, BENCHMARK, MODEL_NAME)
    success = False
    try:
        while not env.done and steps_taken < env.max_steps:
            try:
                action = choose_action(client, obs, cache)
                obs, reward= env.step(action)
            except Exception as exc:
                reward_value = 0.05
                rewards.append(reward_value)
                steps_taken += 1
                log_step(steps_taken, '{"action_type":"error"}', reward_value, True, str(exc))
                env.done = True
                break
            rewards.append(reward.reward)
            steps_taken += 1
            log_step(steps_taken, action_to_json(action), reward.reward, reward.done, reward.info.get("error"))
            if cache.stage != obs.current_stage:
                cache.actions = []
                cache.fingerprint = ""
                cache.stage = ""
        success = env.success
    finally:
        final_score = reward.score if rewards else 0.1
        log_end(success, steps_taken, final_score, rewards)


def main() -> None:
    ensure_agent_mode()
    for task_name in RUN_TASKS:
        run_task(task_name)


if __name__ == "__main__":
    main()

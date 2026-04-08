from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from env.environment import MeetFlowEnv
from env.models import Action


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("MEETFLOW_BENCHMARK", "meetflow_workflow")
MAX_STEPS = int(os.getenv("MAX_STEPS", "14"))
EPISODES_PER_TASK = int(os.getenv("EPISODES_PER_TASK", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.10"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "180"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.70"))
RUN_TASKS = [task.strip() for task in os.getenv("MEETFLOW_TASKS", "easy,medium,hard").split(",") if task.strip()]

TASK_MAX_STEPS = {
    "easy": 14,
    "medium": 18,
    "hard": 34,
}

TASK_SUCCESS_THRESHOLDS = {
    "easy": 0.70,
    "medium": 0.70,
    "hard": 0.70,
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an agent operating a meeting-workflow RL environment.

    Return exactly one JSON object with the keys:
    action_type, item, owner, priority, deadline, dependency, risk_flag

    Allowed action_type values:
    - extract_action_item
    - assign_owner
    - set_priority
    - set_deadline
    - set_dependency
    - mark_risk
    - finalize

    Rules:
    - Respect current_stage and available_actions.
    - Never extract a duplicate or semantically similar item.
    - Complete incomplete extracted items before finalizing.
    - Use short normalized item titles.
    - dependency should be "none" when there is no clear dependency.
    - Never return markdown or explanations.
    """
).strip()


def _field(obj: Any, key: str, default=None):
    if hasattr(obj, key):
        return getattr(obj, key, default)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def to_jsonable(value: Any):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def action_to_str(action: Action) -> str:
    return (
        f"{action.action_type}(item={action.item!r},owner={action.owner!r},priority={action.priority!r},"
        f"deadline={action.deadline!r},dependency={action.dependency!r},risk_flag={action.risk_flag!r})"
    )


def build_user_prompt(obs, task_name: str, step_num: int) -> str:
    meeting_context = to_jsonable(obs.meeting_context)
    extracted_items = to_jsonable(obs.extracted_items)
    unresolved_slots = to_jsonable(obs.unresolved_slots)

    return textwrap.dedent(
        f"""
        Task mode: {task_name}
        Step number: {step_num}

        Transcript:
        {obs.transcript}

        Meeting context:
        {json.dumps(meeting_context, ensure_ascii=False)}

        Current stage: {obs.current_stage}
        Available actions: {json.dumps(obs.available_actions, ensure_ascii=False)}

        Extracted items so far:
        {json.dumps(extracted_items, ensure_ascii=False)}

        Unresolved slots:
        {json.dumps(unresolved_slots, ensure_ascii=False)}

        Notes:
        {json.dumps(obs.notes, ensure_ascii=False)}

        Return one JSON object only.
        """
    ).strip()


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


STOPWORDS = {
    "the", "a", "an", "to", "of", "for", "on", "in", "and", "with",
    "before", "after", "by", "within", "right", "up", "out", "from",
    "today", "tomorrow", "tonight", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "noon", "pm", "am",
    "end", "day", "immediately", "hour", "one", "two",
    "sprint", "planning", "incident", "review", "war", "room",
    "security", "escalation", "launch", "sync", "notes",
}


def _token_set(text: str) -> set:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", _norm(text))
        if token not in STOPWORDS
    }


def _jaccard(a: str, b: str) -> float:
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _seq(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _contains_like(a: str, b: str) -> float:
    na = _norm(a)
    nb = _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.95
    return 0.0


def _similarity(a: str, b: str) -> float:
    return max(_jaccard(a, b), _contains_like(a, b), _seq(a, b) * 0.90)


def _canonicalize_title(text: str) -> str:
    t = _norm(text)
    t = re.sub(
        r"^(standup notes|qa sync|documentation review|incident prep call|customer escalation sync|release gate meeting|launch war room|security escalation|release recovery meeting|incident review|sprint planning):\s*",
        "",
        t,
    )
    t = re.sub(r"\bbecause.*$", "", t)
    t = re.sub(r"\bonce.*$", "", t)
    t = re.sub(r"\bif .*?$", "", t)
    t = re.sub(r"\bafter .*?$", "", t)
    t = re.sub(r"\bbefore\s+\d+\s*(?:pm|am)\b", "", t)
    t = re.sub(r"\bby\s+\d+\s*(?:pm|am)\b", "", t)
    t = re.sub(
        r"\b(today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|immediately|before noon|end of day|within one hour|before rollout|right after)\b",
        "",
        t,
    )
    t = re.sub(r"\s+", " ", t).strip(" ,.-:")
    return t


def _same_item(a: str, b: str) -> bool:
    ca = _canonicalize_title(a)
    cb = _canonicalize_title(b)
    return _similarity(ca, cb) >= 0.76


def _bad_extract_title(title: str) -> bool:
    t = _norm(title)
    if not t:
        return True
    if ":" in t:
        return True
    if len(_token_set(t)) < 2:
        return True
    if len(t.split()) > 10:
        return True
    bad_prefixes = [
        "sprint planning",
        "incident review",
        "launch war room",
        "security escalation",
        "qa sync",
    ]
    if any(t.startswith(prefix) for prefix in bad_prefixes):
        return True
    return False


def _sentence_records(transcript: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    people = ["Priya", "Vikram", "Meena", "Rohit", "Ashu", "Kousik", "Sanjai", "Divya"]

    deadline_patterns = [
        r"before\s+\d+\s*(?:PM|AM)",
        r"by\s+\d+\s*(?:PM|AM)",
        r"within one hour",
        r"right after [^.]+",
        r"after [^.]+",
        r"end of day",
        r"today",
        r"tomorrow",
        r"tonight",
        r"friday",
        r"monday",
        r"wednesday",
        r"thursday",
        r"immediately",
        r"before noon",
        r"before 5 pm",
        r"by 2 pm",
        r"by 3 pm",
        r"before rollout",
        r"if validation fails",
    ]

    action_markers = [
        "will", "should", "must", "needs to", "has to", "need to",
        "action item", "follow up", "owner", "assign", "fix",
        "retest", "refresh", "update", "finish", "validate", "rebuild",
        "hotfix", "coordinate", "review", "send", "prepare", "complete",
        "revoke", "rotate", "approve", "verify",
    ]

    for raw_sentence in re.split(r"[.?!]", transcript):
        sentence = raw_sentence.strip()
        if not sentence:
            continue

        clean_sentence = sentence.strip()
        lowered = clean_sentence.lower()
        if not any(marker in lowered for marker in action_markers):
            continue

        owner = None
        for person in people:
            if re.search(rf"\b{person}\b", clean_sentence):
                owner = person
                break

        title = clean_sentence
        title = re.sub(rf"\b({'|'.join(people)})\b", "", title)
        title = re.sub(
            r"\b(will|should|must|needs to|has to|need to|owns|owns the|will coordinate|has to update|please|kindly)\b",
            " ",
            title,
            flags=re.I,
        )
        title = re.sub(r"\bbecause.*$", "", title, flags=re.I)
        title = re.sub(r"\bonce.*$", "", title, flags=re.I)
        title = re.sub(r"\s+", " ", title).strip(" ,-:")
        title = _canonicalize_title(title)

        if not title or _bad_extract_title(title):
            continue

        found_deadline = None
        for pattern in deadline_patterns:
            match = re.search(pattern, clean_sentence, flags=re.I)
            if match:
                found_deadline = match.group(0)
                break

        priority = "medium"
        if any(w in lowered for w in ["critical", "security", "incident", "compromised", "rollback", "outage", "immediately"]):
            priority = "critical"
        elif any(w in lowered for w in ["blocker", "high priority", "failing", "rebuild", "hotfix", "launch", "escalation", "within one hour", "before rollout"]):
            priority = "high"

        risk_flag = any(
            w in lowered
            for w in [
                "critical", "security", "incident", "compromised", "rollback",
                "outage", "failing", "hotfix", "rebuild", "escalation",
                "revoke", "rotate", "breach", "validation fails",
            ]
        )

        records.append(
            {
                "sentence": clean_sentence,
                "title": title,
                "owner": owner,
                "deadline": found_deadline,
                "priority": priority,
                "risk_flag": risk_flag,
            }
        )

    deduped: List[Dict[str, object]] = []
    for record in records:
        duplicate = False
        for existing in deduped:
            if _same_item(str(record["title"]), str(existing["title"])):
                duplicate = True
                break
        if not duplicate:
            deduped.append(record)

    return deduped


def _record_for_item(item_title: str, records: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    best = None
    best_score = 0.0
    for record in records:
        score = _similarity(_canonicalize_title(item_title), str(record["title"]))
        if score > best_score:
            best_score = score
            best = record
    return best


def _has_similar_item(title: str, items: List[Any]) -> bool:
    for item in items:
        item_title = _field(item, "title", "")
        if _same_item(title, item_title):
            return True
    return False


def _candidate_items(obs) -> List[Dict[str, object]]:
    records = _sentence_records(obs.transcript)
    candidates: List[Tuple[float, Dict[str, object]]] = []

    for record in records:
        title = str(record["title"])
        if not title or _has_similar_item(title, obs.extracted_items):
            continue

        sentence = str(record["sentence"]).lower()
        score = 0.0
        if record.get("owner"):
            score += 1.2
        if record.get("deadline"):
            score += 1.2
        if record.get("priority") == "critical":
            score += 2.0
        elif record.get("priority") == "high":
            score += 1.3
        if record.get("risk_flag"):
            score += 1.5
        if any(
            w in sentence
            for w in [
                "must", "needs", "should", "will", "follow up", "fix", "finish",
                "update", "retest", "validate", "rebuild", "prepare", "revoke",
                "rotate", "approve", "verify",
            ]
        ):
            score += 1.1

        candidates.append((score, record))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in candidates]


def _newest_incomplete_item(items: List[Any]) -> Optional[Any]:
    for item in reversed(items):
        if (
            not _field(item, "owner")
            or not _field(item, "priority")
            or not _field(item, "deadline")
            or _field(item, "dependency") is None
            or _field(item, "risk_flag") is None
        ):
            return item
    return None


def heuristic_action(obs, step_num: int) -> Action:
    items = obs.extracted_items
    stage = obs.current_stage
    records = _sentence_records(obs.transcript)
    candidate_records = _candidate_items(obs)
    target = _newest_incomplete_item(items)

    def infer_priority(item_title: str) -> str:
        record = _record_for_item(item_title, records)
        if record and record.get("priority"):
            return str(record["priority"])
        urgency = _field(obs.meeting_context, "urgency")
        return "high" if urgency in {"high", "critical"} else "medium"

    def infer_risk(item_title: str, priority: Optional[str]) -> bool:
        record = _record_for_item(item_title, records)
        if record is not None:
            return bool(record.get("risk_flag"))
        return priority in {"high", "critical"}

    def infer_deadline(item_title: str) -> str:
        record = _record_for_item(item_title, records)
        if record and record.get("deadline"):
            return str(record["deadline"])
        return "this sprint"

    def infer_owner(item_title: str) -> str:
        record = _record_for_item(item_title, records)
        if record and record.get("owner"):
            return str(record["owner"])
        participants = _field(obs.meeting_context, "participants", []) or ["Unassigned"]
        return participants[0]

    def infer_dependency(item_title: str) -> str:
        record = _record_for_item(item_title, records)
        sentence = str((record or {}).get("sentence") or "").lower()
        if any(token in sentence for token in ["after", "once", "following", "depends on", "if validation fails"]):
            best_other = None
            best_overlap = 0.0
            for other in items:
                other_title = _field(other, "title", "")
                if _same_item(other_title, item_title):
                    continue
                overlap = _jaccard(other_title, sentence)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_other = other_title
            if best_other:
                return best_other
        return "none"

    if stage == "extract_action_item":
        if candidate_records:
            return Action(action_type="extract_action_item", item=str(candidate_records[0]["title"]))
        return Action(action_type="finalize")

    if target is not None:
        target_title = _field(target, "title", "")
        if stage == "assign_owner" and not _field(target, "owner"):
            return Action(action_type="assign_owner", item=target_title, owner=infer_owner(target_title))
        if stage == "set_priority" and not _field(target, "priority"):
            return Action(action_type="set_priority", item=target_title, priority=infer_priority(target_title))
        if stage == "set_deadline" and not _field(target, "deadline"):
            return Action(action_type="set_deadline", item=target_title, deadline=infer_deadline(target_title))
        if stage == "set_dependency" and _field(target, "dependency") is None:
            return Action(action_type="set_dependency", item=target_title, dependency=infer_dependency(target_title))
        if stage == "mark_risk" and _field(target, "risk_flag") is None:
            return Action(
                action_type="mark_risk",
                item=target_title,
                risk_flag=infer_risk(target_title, _field(target, "priority")),
            )

    unresolved = obs.unresolved_slots or []
    if unresolved:
        slot = unresolved[0]
        item_title = _field(slot, "item", "")
        field = _field(slot, "field", "")
        if field == "owner":
            return Action(action_type="assign_owner", item=item_title, owner=infer_owner(item_title))
        if field == "priority":
            return Action(action_type="set_priority", item=item_title, priority=infer_priority(item_title))
        if field == "deadline":
            return Action(action_type="set_deadline", item=item_title, deadline=infer_deadline(item_title))
        if field == "dependency":
            return Action(action_type="set_dependency", item=item_title, dependency=infer_dependency(item_title))
        if field == "risk_flag":
            return Action(action_type="mark_risk", item=item_title, risk_flag=infer_risk(item_title, None))

    if candidate_records and "extract_action_item" in obs.available_actions:
        return Action(action_type="extract_action_item", item=str(candidate_records[0]["title"]))

    return Action(action_type="finalize")


def parse_action(raw_text: str) -> Optional[dict]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return None

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return None

    return None


def _is_valid_model_action(data: dict, obs) -> bool:
    if not isinstance(data, dict):
        return False

    action_type = data.get("action_type")
    if action_type not in set(obs.available_actions):
        return False

    if action_type == "finalize":
        if obs.unresolved_slots or _candidate_items(obs):
            return False

    if action_type in {"assign_owner", "set_priority", "set_deadline", "set_dependency", "mark_risk"}:
        if not str(data.get("item", "")).strip():
            return False

    if action_type == "assign_owner" and not str(data.get("owner", "")).strip():
        return False
    if action_type == "set_priority" and not str(data.get("priority", "")).strip():
        return False
    if action_type == "set_deadline" and not str(data.get("deadline", "")).strip():
        return False
    if action_type == "mark_risk" and data.get("risk_flag") is None:
        return False

    if action_type == "set_dependency":
        dep = str(data.get("dependency", "")).strip().lower()
        if dep and dep != "none":
            existing_titles = [_field(x, "title", "") for x in obs.extracted_items]
            if not any(_same_item(dep, title) for title in existing_titles):
                return False

    return True


def get_model_action(client: OpenAI, obs, task_name: str, step_num: int) -> Action:
    if obs.current_stage == "extract_action_item":
        return heuristic_action(obs, step_num)

    prompt = build_user_prompt(obs, task_name, step_num)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        data = parse_action(raw)
        if _is_valid_model_action(data, obs):
            return Action(**data)
    except Exception:
        pass

    return heuristic_action(obs, step_num)


async def run_single_episode(client: OpenAI, task_name: str, episode_num: int, total_episodes: int) -> Dict:
    task_steps = TASK_MAX_STEPS.get(task_name, MAX_STEPS)
    env = MeetFlowEnv(task_name=task_name, max_steps=task_steps, episode_idx=episode_num)
    rewards: List[float] = []
    steps_taken = 0


    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    obs = env.reset()

    for step in range(1, task_steps + 1):
        action = get_model_action(client, obs, task_name, step)
        obs, reward, done, info = env.step(action)
        reward = max(0.0, min(float(reward), 1.0))
        rewards.append(reward)
        steps_taken = step

        log_step(step, action_to_str(action), reward, done, info.get("last_action_error"))
        if done:
            break

    score = max(0.0, min(float(env.grade()), 1.0))
    success_threshold = TASK_SUCCESS_THRESHOLDS.get(task_name, SUCCESS_SCORE_THRESHOLD)
    success = score >= success_threshold
    log_end(success, steps_taken, score, rewards)

    return {
        "episode": episode_num,
        "task_id": env.task_data.get("id"),
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
    }


async def run_single_task(client: OpenAI, task_name: str) -> Dict:
    episode_results = []
    for episode_num in range(1, EPISODES_PER_TASK + 1):
        episode_result = await run_single_episode(client, task_name, episode_num, EPISODES_PER_TASK)
        episode_results.append(episode_result)

    avg_score = sum(ep["score"] for ep in episode_results) / max(1, len(episode_results))
    avg_steps = sum(ep["steps"] for ep in episode_results) / max(1, len(episode_results))
    success_threshold = TASK_SUCCESS_THRESHOLDS.get(task_name, SUCCESS_SCORE_THRESHOLD)
    success = avg_score >= success_threshold

    return {
        "task": task_name,
        "success": success,
        "episodes": EPISODES_PER_TASK,
        "avg_steps": round(avg_steps, 2),
        "score": avg_score,
        "episode_results": episode_results,
    }


async def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is required.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    results = []

    for task_name in RUN_TASKS:
        results.append(await run_single_task(client, task_name))

    avg_score = sum(result["score"] for result in results) / max(1, len(results))
    print(
        json.dumps(
            {
                "benchmark": BENCHMARK,
                "model": MODEL_NAME,
                "tasks_run": RUN_TASKS,
                "episodes_per_task": EPISODES_PER_TASK,
                "average_score": round(avg_score, 4),
                "results": results,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
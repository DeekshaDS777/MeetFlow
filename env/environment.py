from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from env.grader import compute_reward, final_grade
from env.models import Action, Observation
from env.tasks import get_task


STAGE_ORDER = [
    "extract_action_item",
    "assign_owner",
    "set_priority",
    "set_deadline",
    "set_dependency",
    "mark_risk",
    "finalize",
]

ALLOWED_ACTIONS_BY_STAGE = {
    "extract_action_item": ["extract_action_item"],
    "assign_owner": ["assign_owner", "extract_action_item"],
    "set_priority": ["set_priority", "assign_owner", "extract_action_item"],
    "set_deadline": ["set_deadline", "set_priority", "assign_owner", "extract_action_item"],
    "set_dependency": ["set_dependency", "set_deadline", "set_priority", "assign_owner", "extract_action_item"],
    "mark_risk": ["mark_risk", "set_dependency", "set_deadline", "set_priority", "assign_owner", "extract_action_item"],
    "finalize": ["finalize", "extract_action_item", "mark_risk", "set_dependency", "set_deadline", "set_priority", "assign_owner"],
}

SUCCESS_THRESHOLD = 0.58
AUTO_STOP_HIGH_SCORE = 0.92
ITEM_MATCH_THRESHOLD = 0.52


class MeetFlowEnv:
    def __init__(self, task_name: str = "hard", max_steps: int = 14, episode_idx: Optional[int] = None):
        self.task_name = task_name
        self.max_steps = max_steps
        self.episode_idx = episode_idx
        self.reset()

    @staticmethod
    def _norm(text: Optional[str]) -> str:
        return (text or "").strip().lower()

    def _sim(self, a: str, b: str) -> float:
        return SequenceMatcher(None, self._norm(a), self._norm(b)).ratio()

    def _find_item(self, item_name: str) -> Optional[Dict]:
        best = None
        best_sim = 0.0
        for item in self.predictions:
            sim = self._sim(item.get("title", ""), item_name)
            if sim > best_sim:
                best = item
                best_sim = sim
        if best is not None and best_sim >= ITEM_MATCH_THRESHOLD:
            return best
        return None

    def _upsert_item(self, item_name: str) -> Dict:
        found = self._find_item(item_name)
        if found is not None:
            return found

        new_item = {
            "title": item_name.strip(),
            "owner": None,
            "priority": None,
            "deadline": None,
            "dependency": None,
            "risk_flag": None,
            "status": "open",
        }
        self.predictions.append(new_item)
        return new_item

    def _derive_stage_index(self) -> int:
        if not self.predictions:
            return 0

        if any(item.get("owner") is None for item in self.predictions):
            return 1
        if any(item.get("priority") is None for item in self.predictions):
            return 2
        if any(item.get("deadline") is None for item in self.predictions):
            return 3
        if any(item.get("dependency") is None for item in self.predictions):
            return 4
        if any(item.get("risk_flag") is None for item in self.predictions):
            return 5
        return 6

    def _unresolved_slots(self) -> List[Dict]:
        unresolved = []
        for item in self.predictions:
            for field in ["owner", "priority", "deadline", "dependency", "risk_flag"]:
                if item.get(field) is None:
                    unresolved.append({"item": item.get("title", ""), "field": field})
        return unresolved

    def _notes(self) -> List[str]:
        notes: List[str] = []
        current_stage = STAGE_ORDER[self.stage_index]

        if current_stage != "finalize":
            notes.append(f"Current stage expects actions for: {current_stage}")

        if self.last_action_error:
            notes.append(f"Last action issue: {self.last_action_error}")

        if self.steps >= self.max_steps - 2:
            notes.append("Episode is near the step limit. Prefer consolidating remaining fields.")

        if self.success:
            notes.append("Success threshold reached, but continue improving coverage before finalize.")

        return notes

    def _observation(self) -> Observation:
        return Observation(
            transcript=self.transcript,
            current_stage=STAGE_ORDER[self.stage_index],
            stage_index=self.stage_index,
            step_count=self.steps,
            available_actions=ALLOWED_ACTIONS_BY_STAGE[STAGE_ORDER[self.stage_index]],
            extracted_items=deepcopy(self.predictions),
            meeting_context=deepcopy(self.meeting_context),
            unresolved_slots=self._unresolved_slots(),
            notes=self._notes(),
        )

    def reset(self, task_data=None):
        task = task_data if task_data is not None else get_task(self.task_name, self.episode_idx)

        self.task_data = task
        self.transcript = task["transcript"]
        self.gt = deepcopy(task["ground_truth"])
        self.meeting_context = deepcopy(task.get("meeting_context", {}))

        self.predictions: List[Dict] = []
        self.steps = 0
        self.done = False
        self.success = False
        self.last_action_error: Optional[str] = None
        self.stage_index = 0

        return self._observation()

    def _validate_stage(self, action_type: str) -> Optional[str]:
        allowed = ALLOWED_ACTIONS_BY_STAGE[STAGE_ORDER[self.stage_index]]
        if action_type not in allowed:
            return f"action {action_type} is not allowed in stage {STAGE_ORDER[self.stage_index]}"
        return None

    def step(self, action: Action):
        if self.done:
            return self._observation(), 0.0, True, {"last_action_error": "episode already finished"}

        prev_predictions = deepcopy(self.predictions)
        self.steps += 1
        self.last_action_error = self._validate_stage(action.action_type)

        if not self.last_action_error:
            if action.action_type == "extract_action_item":
                if not action.item.strip():
                    self.last_action_error = "missing item title"
                else:
                    self._upsert_item(action.item)

            elif action.action_type == "assign_owner":
                if not action.item.strip() or not action.owner:
                    self.last_action_error = "assign_owner needs item and owner"
                else:
                    item = self._upsert_item(action.item)
                    item["owner"] = action.owner

            elif action.action_type == "set_priority":
                if not action.item.strip() or not action.priority:
                    self.last_action_error = "set_priority needs item and priority"
                else:
                    item = self._upsert_item(action.item)
                    item["priority"] = action.priority

            elif action.action_type == "set_deadline":
                if not action.item.strip() or not action.deadline:
                    self.last_action_error = "set_deadline needs item and deadline"
                else:
                    item = self._upsert_item(action.item)
                    item["deadline"] = action.deadline

            elif action.action_type == "set_dependency":
                if not action.item.strip():
                    self.last_action_error = "set_dependency needs item"
                else:
                    item = self._upsert_item(action.item)
                    item["dependency"] = action.dependency or "none"

            elif action.action_type == "mark_risk":
                if not action.item.strip() or action.risk_flag is None:
                    self.last_action_error = "mark_risk needs item and risk_flag"
                else:
                    item = self._upsert_item(action.item)
                    item["risk_flag"] = action.risk_flag

            elif action.action_type == "finalize":
                score_now = final_grade(self.predictions, self.gt)
                unresolved = self._unresolved_slots()

                if self.steps <= 3 and score_now < 0.50:
                    self.last_action_error = "premature finalize"
                elif unresolved and score_now < 0.80:
                    self.last_action_error = "finalize before filling important fields"
                else:
                    self.done = True

        self.stage_index = self._derive_stage_index()

        reward = compute_reward(
            prev_predictions,
            self.predictions,
            self.gt,
            action.action_type,
            self.last_action_error,
        )

        current_score = final_grade(self.predictions, self.gt)

        if current_score >= SUCCESS_THRESHOLD:
            self.success = True

        # Do not stop immediately at success threshold.
        # Let the agent continue until explicit finalize or very high score.
        if current_score >= AUTO_STOP_HIGH_SCORE:
            self.done = True

        if self.steps >= self.max_steps:
            self.done = True

        obs = self._observation()
        return obs, float(min(1.0, max(0.0, reward))), self.done, {"last_action_error": self.last_action_error}

    def state(self):
        return {
            "task_name": self.task_name,
            "task_id": self.task_data.get("id"),
            "episode_idx": self.episode_idx,
            "transcript": self.transcript,
            "meeting_context": deepcopy(self.meeting_context),
            "predictions": deepcopy(self.predictions),
            "ground_truth_size": len(self.gt),
            "steps": self.steps,
            "done": self.done,
            "success": self.success,
            "stage": STAGE_ORDER[self.stage_index],
            "last_action_error": self.last_action_error,
            "score": self.grade(),
        }

    def grade(self) -> float:
        return float(final_grade(self.predictions, self.gt))
from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
from typing import List, Optional

from models import Action, ActionItem, EnvironmentState, MeetingContext, Observation, Reward, UnresolvedSlot
from server.services.grader import step_reward, structured_score
from server.services.task_loader import load_task_bank

STAGE_ORDER = [
    "extract_action_item",
    "assign_owner",
    "set_priority",
    "set_deadline",
    "set_dependency",
    "set_risk_flag",
    "finalize",
]

SUCCESS_THRESHOLDS = {"easy": 0.68, "medium": 0.70, "hard": 0.72}
TASK_BANK = load_task_bank()


class MeetFlowEnvironment:
    def __init__(self, task_name: str = "easy", episode_index: int = 0, max_steps: Optional[int] = None):
        self.task_name = task_name
        self.episode_index = episode_index
        self.max_steps = max_steps or {"easy": 18, "medium": 22, "hard": 36}[task_name]
        self.reset()

    @staticmethod
    def _norm(text: Optional[str]) -> str:
        return (text or "").strip().lower()

    def _sim(self, a: str, b: str) -> float:
        return SequenceMatcher(None, self._norm(a), self._norm(b)).ratio()

    def _current_stage(self) -> str:
        if len(self.predictions) < len(self.truth_items):
            return "extract_action_item"
        if any(item.owner is None for item in self.predictions):
            return "assign_owner"
        if any(item.priority is None for item in self.predictions):
            return "set_priority"
        if any(item.deadline is None for item in self.predictions):
            return "set_deadline"
        if any(item.dependency is None for item in self.predictions):
            return "set_dependency"
        if any(item.risk_flag is None for item in self.predictions):
            return "set_risk_flag"
        return "finalize"

    def _stage_index(self) -> int:
        return STAGE_ORDER.index(self._current_stage())

    def _available_actions(self) -> List[str]:
        return [self._current_stage()]

    def _find_prediction(self, item_name: str) -> Optional[ActionItem]:
        best_item: Optional[ActionItem] = None
        best_score = 0.0
        for item in self.predictions:
            score = self._sim(item.title, item_name)
            if score > best_score:
                best_item, best_score = item, score
        return best_item if best_item and best_score >= 0.60 else None

    def _duplicate_title(self, title: str) -> bool:
        return any(self._sim(existing.title, title) >= 0.82 for existing in self.predictions)

    def _unresolved_slots(self) -> List[UnresolvedSlot]:
        unresolved: List[UnresolvedSlot] = []
        for item in self.predictions:
            if item.owner is None:
                unresolved.append(UnresolvedSlot(item=item.title, field="owner"))
            if item.priority is None:
                unresolved.append(UnresolvedSlot(item=item.title, field="priority"))
            if item.deadline is None:
                unresolved.append(UnresolvedSlot(item=item.title, field="deadline"))
            if item.dependency is None:
                unresolved.append(UnresolvedSlot(item=item.title, field="dependency"))
            if item.risk_flag is None:
                unresolved.append(UnresolvedSlot(item=item.title, field="risk_flag"))
        return unresolved

    def _observation(self) -> Observation:
        notes = []
        if self.last_action_error:
            notes.append(f"Last action issue: {self.last_action_error}")
        if self.steps >= self.max_steps - 3:
            notes.append("Episode is near the step limit.")
        if self.repeat_count >= 2:
            notes.append("Avoid repeating the same action; progress to a new unresolved item.")
        if self._current_stage() == "extract_action_item":
            notes.append("Extraction should create one unique normalized action item title per step.")
        return Observation(
            transcript=self.transcript,
            current_stage=self._current_stage(),
            stage_index=self._stage_index(),
            step_count=self.steps,
            available_actions=self._available_actions(),
            extracted_items=deepcopy(self.predictions),
            meeting_context=self.meeting_context,
            unresolved_slots=self._unresolved_slots(),
            notes=notes,
        )

    def _state(self) -> EnvironmentState:
        return EnvironmentState(
            session_id=self.session_id,
            task_name=self.task_name,
            task_id=self.task_id,
            transcript=self.transcript,
            meeting_context=self.meeting_context,
            predictions=deepcopy(self.predictions),
            ground_truth_size=len(self.truth_items),
            steps=self.steps,
            done=self.done,
            success=self.success,
            stage=self._current_stage(),
            last_action_error=self.last_action_error,
            score=self.score,
        )

    def reset(self, session_id: Optional[str] = None) -> Observation:
        tasks = TASK_BANK[self.task_name]
        record = tasks[self.episode_index % len(tasks)]
        self.session_id = session_id
        self.task_id = record["id"]
        self.transcript = record["transcript"]
        self.meeting_context = MeetingContext.model_validate(record["meeting_context"])
        self.truth_items = [ActionItem.model_validate(item) for item in record["action_items"]]
        self.predictions: List[ActionItem] = []
        self.steps = 0
        self.done = False
        self.success = False
        self.last_action_error: Optional[str] = None
        self.score = 0.01
        self.last_signature: Optional[str] = None
        self.repeat_count = 0
        return self._observation()

    def _validate_stage(self, action: Action) -> bool:
        return action.action_type == self._current_stage()

    def _signature(self, action: Action) -> str:
        import json
        return json.dumps(action.model_dump(), sort_keys=True)

    def _update_existing(self, action: Action, field_name: str, value) -> None:
        existing = self._find_prediction(action.item)
        if existing is None:
            raise ValueError("item must already exist before filling later fields")
        setattr(existing, field_name, value)

    def _apply_action(self, action: Action) -> None:
        if action.action_type == "extract_action_item":
            if not action.item:
                raise ValueError("extract_action_item requires a non-empty item")
            if self._duplicate_title(action.item):
                raise ValueError("duplicate or semantically similar item")
            self.predictions.append(ActionItem(title=action.item))
            return
        if action.action_type == "assign_owner":
            if not action.owner:
                raise ValueError("assign_owner requires owner")
            self._update_existing(action, "owner", action.owner)
            return
        if action.action_type == "set_priority":
            if not action.priority:
                raise ValueError("set_priority requires priority")
            self._update_existing(action, "priority", action.priority)
            return
        if action.action_type == "set_deadline":
            if not action.deadline:
                raise ValueError("set_deadline requires deadline")
            self._update_existing(action, "deadline", action.deadline)
            return
        if action.action_type == "set_dependency":
            if action.dependency is None:
                raise ValueError("set_dependency requires dependency")
            self._update_existing(action, "dependency", action.dependency)
            return
        if action.action_type == "set_risk_flag":
            if action.risk_flag is None:
                raise ValueError("set_risk_flag requires risk_flag")
            self._update_existing(action, "risk_flag", action.risk_flag)
            return
        if action.action_type == "finalize":
            if self._current_stage() != "finalize":
                raise ValueError("cannot finalize before all required fields are filled")
            self.done = True
            return
        raise ValueError("unsupported action_type")

    def step(self, action: Action) -> tuple[Observation, Reward]:
        if self.done:
            obs = self._observation()
            return obs, Reward(reward=0.01, done=True, score=self.score, info={"error": "episode already completed"})
        self.steps += 1
        previous = self.score
        signature = self._signature(action)
        repeated = signature == self.last_signature
        self.repeat_count = self.repeat_count + 1 if repeated else 0
        valid_action = self._validate_stage(action)
        self.last_signature = signature
        try:
            if not valid_action:
                raise ValueError(f"expected action_type {self._current_stage()} but received {action.action_type}")
            self._apply_action(action)
            self.last_action_error = None
        except Exception as exc:
            self.last_action_error = str(exc)
        scoring = structured_score(self.predictions, self.truth_items, self.task_name)
        self.score = scoring["score"]
        stage_complete = self._current_stage() == "finalize"
        if self.repeat_count >= 3:
            self.done = True
            self.last_action_error = self.last_action_error or "repeated action loop detected"
        elif self.done:
            self.success = stage_complete and self.score >= SUCCESS_THRESHOLDS[self.task_name]
        elif self.steps >= self.max_steps:
            self.done = True
            self.success = stage_complete and self.score >= SUCCESS_THRESHOLDS[self.task_name]
        if self.done and not self.success:
            self.success = stage_complete and self.score >= SUCCESS_THRESHOLDS[self.task_name]
        reward_value = step_reward(
            previous_score=previous,
            current_score=self.score,
            valid_action=self.last_action_error is None,
            difficulty=self.task_name,
            stage_index=self._stage_index(),
            step_count=self.steps,
            repeated=repeated,
        )
        obs = self._observation()
        reward = Reward(
            reward=reward_value,
            done=self.done,
            score=self.score,
            info={
                "success": self.success,
                "error": self.last_action_error,
                "coverage": scoring["coverage"],
                "precision": scoring["precision"],
            },
        )
        return obs, reward

    def state(self) -> EnvironmentState:
        return self._state()

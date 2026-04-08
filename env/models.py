from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


StageLiteral = Literal[
    "extract_action_item",
    "assign_owner",
    "set_priority",
    "set_deadline",
    "set_dependency",
    "mark_risk",
    "finalize",
]

PriorityLiteral = Literal["low", "medium", "high", "critical"]
StatusLiteral = Literal["open", "in_progress", "blocked", "done"]


def _normalize_priority(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip().lower()
    aliases = {
        "normal": "medium",
        "urgent": "critical",
        "p1": "critical",
        "p2": "high",
        "p3": "medium",
    }
    return aliases.get(value, value)


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, description="Normalized action item title")
    owner: Optional[str] = Field(default=None, description="Assigned owner")
    priority: Optional[PriorityLiteral] = Field(default=None, description="Priority level")
    deadline: Optional[str] = Field(default=None, description="Natural language deadline")
    dependency: Optional[str] = Field(default=None, description="Dependency title or 'none'")
    risk_flag: Optional[bool] = Field(default=None, description="Whether this task is operationally risky")
    status: StatusLiteral = Field(default="open", description="Workflow status")

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be empty")
        return value

    @field_validator("owner", "deadline", "dependency")
    @classmethod
    def normalize_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_priority(value)


class UnresolvedSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str = Field(..., min_length=1, description="Action item title")
    field: Literal["owner", "priority", "deadline", "dependency", "risk_flag"]


class MeetingContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    meeting_type: Optional[str] = None
    team: Optional[str] = None
    urgency: Optional[PriorityLiteral] = None
    participants: List[str] = Field(default_factory=list)
    owner_hint: Optional[str] = None

    @field_validator("urgency", mode="before")
    @classmethod
    def normalize_urgency(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_priority(value)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(..., description="Meeting transcript")
    current_stage: StageLiteral = Field(..., description="Current workflow stage")
    stage_index: int = Field(..., ge=0, description="Stage index in the workflow")
    step_count: int = Field(..., ge=0, description="Current step number")
    available_actions: List[StageLiteral] = Field(default_factory=list)
    extracted_items: List[ActionItem] = Field(default_factory=list)
    meeting_context: MeetingContext = Field(default_factory=MeetingContext)
    unresolved_slots: List[UnresolvedSlot] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @field_validator("transcript")
    @classmethod
    def validate_transcript(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("transcript must not be empty")
        return value


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: StageLiteral
    item: str = ""
    owner: Optional[str] = None
    priority: Optional[PriorityLiteral] = None
    deadline: Optional[str] = None
    dependency: Optional[str] = None
    risk_flag: Optional[bool] = None

    @field_validator("item")
    @classmethod
    def normalize_item(cls, value: str) -> str:
        return value.strip()

    @field_validator("owner", "deadline", "dependency")
    @classmethod
    def normalize_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_priority(value)


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: float = Field(..., ge=0.0, le=1.0, description="Step reward")
    done: bool = Field(..., description="Whether the episode has ended")
    score: float = Field(..., ge=0.0, le=1.0, description="Current cumulative score")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional environment info")


class EnvironmentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str
    task_id: Optional[str] = None
    transcript: str
    meeting_context: MeetingContext = Field(default_factory=MeetingContext)
    predictions: List[ActionItem] = Field(default_factory=list)
    ground_truth_size: int = Field(..., ge=0)
    steps: int = Field(..., ge=0)
    done: bool
    success: bool
    stage: StageLiteral
    last_action_error: Optional[str] = None
    score: float = Field(..., ge=0.0, le=1.0)
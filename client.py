from __future__ import annotations

from typing import Optional

import httpx

from models import Action, EnvironmentState, Observation, Reward


class MeetFlowClient:
    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = httpx.Client(timeout=timeout)
        self.session_id: Optional[str] = None

    def reset(self, task_name: str = "easy") -> Observation:
        response = self.session.post(f"{self.base_url}/reset", params={"task_name": task_name})
        response.raise_for_status()
        payload = response.json()
        self.session_id = payload["session_id"]
        return Observation.model_validate(payload["observation"])

    def step(self, action: Action) -> Reward:
        if not self.session_id:
            raise RuntimeError("Call reset() before step().")
        response = self.session.post(
            f"{self.base_url}/step",
            json={"session_id": self.session_id, "action": action.model_dump()},
        )
        response.raise_for_status()
        payload = response.json()
        return Reward.model_validate(payload["reward"])

    def state(self) -> EnvironmentState:
        if not self.session_id:
            raise RuntimeError("Call reset() before state().")
        response = self.session.get(f"{self.base_url}/state", params={"session_id": self.session_id})
        response.raise_for_status()
        return EnvironmentState.model_validate(response.json()["state"])

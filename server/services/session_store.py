from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict
from uuid import uuid4

from server.environment import MeetFlowEnvironment


@dataclass
class SessionRecord:
    env: MeetFlowEnvironment


class SessionStore:
    def __init__(self) -> None:
        self._data: Dict[str, SessionRecord] = {}
        self._lock = Lock()

    def create(self, task_name: str) -> tuple[str, MeetFlowEnvironment]:
        env = MeetFlowEnvironment(task_name=task_name)
        session_id = uuid4().hex
        with self._lock:
            self._data[session_id] = SessionRecord(env=env)
        return session_id, env

    def get(self, session_id: str) -> MeetFlowEnvironment:
        with self._lock:
            record = self._data[session_id]
            return record.env

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

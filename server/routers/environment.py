from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from models import Action
from server.services.session_store import SessionStore

router = APIRouter()
store = SessionStore()


class StepRequest(BaseModel):
    session_id: str
    action: Action


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/reset")
def reset(task_name: str = Query(default="easy", pattern="^(easy|medium|hard)$")) -> dict:
    session_id, env = store.create(task_name=task_name)
    env.session_id = session_id
    observation = env.reset(session_id=session_id)
    return {"session_id": session_id, "observation": observation.model_dump(), "state": env.state().model_dump()}


@router.post("/step")
def step(payload: StepRequest) -> dict:
    try:
        env = store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id") from exc
    observation, reward = env.step(payload.action)
    return {"observation": observation.model_dump(), "reward": reward.model_dump(), "state": env.state().model_dump()}


@router.get("/state")
def state(session_id: str) -> dict:
    try:
        env = store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id") from exc
    return {"state": env.state().model_dump()}

from __future__ import annotations

from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models import Action
from server.environment import MeetFlowEnvironment

APP_TITLE = "MeetFlow"
APP_VERSION = "3.0.3"
DEFAULT_TASK = "hard"
HOST = "0.0.0.0"
PORT = 7860

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Single active environment for hackathon submission flow
env = MeetFlowEnvironment(task_name=DEFAULT_TASK)


class ResetRequest(BaseModel):
    task_name: Optional[str] = Field(default=DEFAULT_TASK)


def _serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": APP_TITLE,
        "status": "ok",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> dict[str, Any]:
    global env
    try:
        task_name = DEFAULT_TASK
        if request is not None and request.task_name:
            task_name = str(request.task_name).strip() or DEFAULT_TASK

        env = MeetFlowEnvironment(task_name=task_name)
        observation = env.reset()

        return {
            "observation": _serialize(observation),
            "task_name": env.task_name,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"reset failed: {exc}") from exc


@app.post("/step")
def step(action: Action) -> dict[str, Any]:
    try:
        observation, reward = env.step(action)

        score = reward.reward
        if isinstance(reward.info, dict) and "score" in reward.info:
            score = reward.info["score"]

        return {
            "observation": _serialize(observation),
            "reward": float(f"{float(reward.reward):.4f}"),
            "done": bool(reward.done),
            "info": reward.info if isinstance(reward.info, dict) else {"info": str(reward.info)},
            "score": float(f"{float(score):.4f}"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"step failed: {exc}") from exc


@app.get("/state")
def state() -> Any:
    try:
        state_obj = env.state()
        return _serialize(state_obj)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"state failed: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "task_name": str(env.task_name),
    }


def main() -> None:
    uvicorn.run("server.app:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
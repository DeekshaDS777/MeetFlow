from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from env.environment import MeetFlowEnv
from env.models import Action

app = FastAPI(title="MeetFlow", version="2.0.0")
env = MeetFlowEnv(task_name="hard")


class ResetRequest(BaseModel):
    task_name: Optional[str] = "hard"


@app.get("/")
def root():
    return {
        "name": "MeetFlow",
        "status": "ok",
        "version": "2.0.0",
    }


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    task_name = "hard"

    if request is not None and request.task_name:
        task_name = request.task_name

    env.task_name = task_name
    observation = env.reset()

    return {
        "observation": observation.model_dump(),
        "task_name": env.task_name,
    }


@app.post("/step")
def step(action: Action):
    observation, reward, done, info = env.step(action)

    score = reward
    if isinstance(info, dict) and "score" in info:
        score = info["score"]

    return {
        "observation": observation.model_dump(),
        "reward": float(f"{reward:.4f}"),
        "done": done,
        "info": info,
        "score": float(f"{score:.4f}"),
    }


@app.get("/state")
def state():
    state_obj = env.state()
    return state_obj.model_dump() if hasattr(state_obj, "model_dump") else state_obj


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "task_name": env.task_name,
    }


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
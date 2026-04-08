from fastapi import FastAPI

from env.environment import MeetFlowEnv
from env.models import Action

app = FastAPI(title="MeetFlow", version="2.0.0")
env = MeetFlowEnv(task_name="hard")


@app.get("/")
def root():
    return {"name": "MeetFlow", "status": "ok", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "task_name": env.task_name}


@app.post("/reset")
def reset(task_name: str = "hard"):
    env.task_name = task_name
    return env.reset().model_dump()


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(f"{reward:.4f}"),
        "done": done,
        "info": info,
        "score": float(f"{env.grade():.4f}"),
    }


@app.get("/state")
def state():
    return env.state()

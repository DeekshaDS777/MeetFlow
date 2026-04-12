# MeetFlow Phase 3 Ready Repository

MeetFlow is a real-world, multi-stage OpenEnv-style environment for converting messy meeting transcripts into structured operational action items. It is built for strict evaluation and hybrid agent baselines: deterministic extraction followed by LLM-driven downstream decisions.

## Why this benchmark is useful

Real organizations do not receive neatly structured task lists. They receive meeting transcripts, escalation notes, release gate discussions, incident reviews, and launch war room updates. MeetFlow measures whether an agent can:

- discover all action items without duplicates,
- assign correct owners,
- infer priority and urgency,
- resolve deadlines and dependencies,
- flag operational risk,
- finalize only when the workflow is complete.

This is more realistic than a single-step classifier because the agent must take a sequence of correct decisions under state constraints.

## Repository structure

```text
project-root/
├── inference.py
├── openenv.yaml
├── models.py
├── client.py
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── README.md
├── server/
│   ├── app.py
│   ├── main.py
│   ├── environment.py
│   ├── routers/
│   │   └── environment.py
│   └── services/
│       ├── grader.py
│       ├── session_store.py
│       └── task_loader.py
└── tasks/
    ├── easy.json
    ├── medium.json
    └── hard.json
```

## Architecture

### Environment state machine

Stages:
1. `extract_action_item`
2. `assign_owner`
3. `set_priority`
4. `set_deadline`
5. `set_dependency`
6. `set_risk_flag`
7. `finalize`

The environment does **not** allow later-stage fields to create items silently. That closes a major reward-hacking loophole.

### Observation schema

Each observation returns:
- transcript
- current_stage
- stage_index
- step_count
- available_actions
- extracted_items
- meeting_context
- unresolved_slots
- notes

### Action schema

Each action must contain:
- `action_type`
- `item`
- `owner`
- `priority`
- `deadline`
- `dependency`
- `risk_flag`

### Hybrid agent design

The baseline agent is intentionally hybrid:
- **Heuristic** for `extract_action_item`
- **LLM (OpenAI client only)** for all remaining stages

This gives stable duplicate-safe extraction while keeping downstream decision making genuinely model-driven.

Environment variables used by inference:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

## Tasks

### Easy
Routine planning and documentation tasks with direct owners and explicit deadlines.

### Medium
Operational tasks with stronger urgency and dependency chains.

### Hard
Security and launch workflows with dependencies, risk, and multiple interlocked actions.

## Reward design

The structured scorer uses weighted field scoring:
- title: 0.22
- owner: 0.18
- priority: 0.16
- deadline: 0.16
- dependency: 0.12
- risk flag: 0.16

The final score combines:
- normalized field accuracy,
- coverage,
- precision,
- a small difficulty bonus.

Step rewards are shaped from score improvement, validity, difficulty, stage, and repetition penalties. Rewards are clamped to the open interval `(0.01, 0.99)`.

## Strict inference output

The inference script emits exactly:
```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Run the hybrid inference baseline:

```bash
export HF_TOKEN=your_hf_token
python inference.py
```

## Hugging Face Spaces deployment

1. Create a Docker Space.
2. Push this repository.
3. Add secrets:
   - `HF_TOKEN`
   - `API_BASE_URL` (optional, defaults to HF router)
   - `MODEL_NAME` (optional)
4. Build and confirm `/health` and `/reset` respond.


## Validation checklist

Before submission:

```bash
python -m py_compile inference.py models.py client.py server/app.py server/main.py server/environment.py server/routers/environment.py server/services/grader.py server/services/session_store.py server/services/task_loader.py
openenv validate
docker build -t meetflow-phase3 .
python inference.py
```

Expected behavior:
- `/reset` returns `session_id`, `observation`, and `state`
- `/step` enforces stage-correct actions only
- `/state` returns typed `EnvironmentState`
- rewards and final scores stay strictly inside `(0.01, 0.99)`
- the hybrid baseline uses heuristic extraction and LLM decisions for every later stage

## Sample output

```text
[START] task=easy env=meetflow_workflow model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"extract_action_item","item":"refresh onboarding guide","owner":null,"priority":null,"deadline":null,"dependency":null,"risk_flag":null} reward=0.16 done=false error=null
[STEP] step=2 action={"action_type":"extract_action_item","item":"prepare customer demo checklist","owner":null,"priority":null,"deadline":null,"dependency":null,"risk_flag":null} reward=0.20 done=false error=null
[STEP] step=3 action={"action_type":"extract_action_item","item":"verify production backup","owner":null,"priority":null,"deadline":null,"dependency":null,"risk_flag":null} reward=0.23 done=false error=null
[END] success=false steps=3 rewards=0.16,0.20,0.23
```

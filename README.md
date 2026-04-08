# MeetFlow

MeetFlow is a multi-step OpenEnv-style environment for **meeting workflow automation**.  
It simulates a real-world task where an agent must read messy operational transcripts and convert them into structured, actionable work items.

The environment is designed for evaluating and training agents that can:

- extract action items from meeting transcripts
- assign owners
- set priorities
- capture deadlines
- resolve dependencies
- mark operational risk
- decide when the workflow is complete

---

## Why this environment matters

In real organizations, important work often begins in meetings:

- incident reviews
- launch war rooms
- release gates
- customer escalations
- security response calls

The output of these meetings is usually unstructured. MeetFlow turns that into a benchmarkable RL-style environment where an agent must progressively structure the work, rather than solving a one-shot classification problem.

This makes the environment useful for:

- agent evaluation
- workflow automation research
- structured reasoning benchmarks
- long-horizon decision making

---

## Environment design

MeetFlow is a **multi-stage environment** with the following workflow stages:

1. `extract_action_item`
2. `assign_owner`
3. `set_priority`
4. `set_deadline`
5. `set_dependency`
6. `mark_risk`
7. `finalize`

The environment tracks incomplete fields and dynamically moves the agent through these stages.

### Observation space

Each observation contains:

- `transcript`: raw meeting transcript
- `current_stage`: active workflow stage
- `stage_index`: numeric stage index
- `step_count`: current step number
- `available_actions`: actions allowed at the current stage
- `extracted_items`: structured action items collected so far
- `meeting_context`: meeting metadata such as urgency, team, and participants
- `unresolved_slots`: missing fields that still need to be filled
- `notes`: environment hints about state and progression

### Action space

Each action contains:

- `action_type`
- `item`
- `owner`
- `priority`
- `deadline`
- `dependency`
- `risk_flag`

Supported `action_type` values:

- `extract_action_item`
- `assign_owner`
- `set_priority`
- `set_deadline`
- `set_dependency`
- `mark_risk`
- `finalize`

---

## Tasks

MeetFlow provides three difficulty levels:

### Easy
Routine coordination tasks with straightforward owners and deadlines.

### Medium
Operational tasks with stronger urgency and dependency chains.

### Hard
Security, launch, and recovery workflows with multiple interacting tasks, dependency ordering, and risk-heavy decisions.

The task bank contains multiple scenarios per difficulty, and the inference script can run multiple episodes per task.

---

## Reward and grading

MeetFlow uses a deterministic structured grader with scores in the range **0.0 to 1.0**.

The grader rewards:

- semantic title matching
- correct owner assignment
- correct priority
- deadline accuracy
- dependency accuracy
- correct risk marking
- completion coverage

It also applies:

- partial credit for near matches
- coverage bonus for broader task completion
- small precision penalty for noisy over-prediction

The reward function is shaped from **score improvement over time**, so the agent receives useful trajectory-level signal instead of only a final binary result.

---

## Hybrid policy used in baseline inference

The baseline agent uses a **hybrid policy**:

- **heuristic extraction** for `extract_action_item`
- **LLM-first reasoning** for all remaining stages
- heuristic fallback if the LLM output is invalid

This split is intentional:
- extraction benefits from deterministic duplicate-safe logic
- later stages benefit from LLM reasoning over ownership, priority, deadlines, dependencies, and risk

---

## API endpoints

- `POST /reset?task_name=hard`
- `POST /step`
- `GET /state`
- `GET /health`

---

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
# MeetFlow

MeetFlow is a multi-step OpenEnv-style environment for meeting workflow automation.

## What it does
- extracts action items from messy meeting transcripts
- assigns owners
- sets priority
- captures deadlines
- links dependencies
- marks operational risk
- produces a final grade in the 0.0–1.0 range

## Tasks
- easy
- medium
- hard

## Endpoints
- `POST /reset?task_name=hard`
- `POST /step`
- `GET /state`
- `GET /health`

## Run locally
```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

## Run inference
Set:
- `HF_TOKEN`
- optional `API_BASE_URL`
- optional `MODEL_NAME`

Then run:
```bash
python inference.py
```

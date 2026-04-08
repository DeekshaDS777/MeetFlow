---
title: MeetFlow OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# 🚀 MeetFlow OpenEnv Environment

MeetFlow is a multi-step workflow automation environment built using OpenEnv principles. It simulates meeting operations and evaluates structured decision-making across multiple stages.

---

## 🔥 Features

- ✅ OpenEnv compliant environment  
- ✅ Multi-step workflow reasoning  
- ✅ 3 difficulty levels:
  - Easy
  - Medium
  - Hard  
- ✅ Deterministic structured grader  
- ✅ Reward range: 0.0 – 1.0  
- ✅ Fully Dockerized deployment  

---

## 🧠 Environment Overview

The environment simulates a meeting workflow where an agent must:

- Extract structured tasks from transcripts  
- Assign ownership  
- Set deadlines  
- Track workflow completion  

Each step contributes to a final reward score.

---

## ⚙️ API Endpoints

| Endpoint | Method | Description |
|--------|--------|------------|
| `/reset` | POST | Initialize environment |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |

---

## 📦 Example Usage

### Reset
```bash
curl -X POST /reset \
-H "Content-Type: application/json" \
-d '{"task_name":"easy"}'
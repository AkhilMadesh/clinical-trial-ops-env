---
title: ClinicalTrialOps-Env
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
license: mit
tags:
  - openenv
  - healthcare
  - clinical-trials
  - reinforcement-learning
  - agent-evaluation
  - pharma
  - real-world
short_description: OpenEnv agent environment — clinical trial coordinator AI
---
# ClinicalTrialOps-Env

**OpenEnv-compliant environment** — AI agent as clinical trial coordinator.
Phase II NSCLC oncology study (fictional protocol: LUMINOS-201, drug: Velitinib).

[![openenv](https://img.shields.io/badge/openenv-compliant-green)](https://github.com/openenv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)

---

## Live API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/tasks` | List all 3 tasks |
| POST | `/reset` | Start episode |
| POST | `/step` | Submit action |
| GET | `/state` | Current state |
| GET | `/docs` | Swagger UI |

---

## 3 Tasks

| Task | Difficulty | Baseline Score |
|------|-----------|----------------|
| eligibility_screening | Easy | ~0.86 |
| lab_dose_management | Medium | ~0.59 |
| sae_triage | Hard | ~0.36 |

---

## Quick Start

```python
import requests
BASE = "https://YOUR_USERNAME-clinicaltrialops-env.hf.space"
obs = requests.post(f"{BASE}/reset",
    json={"task_id": "eligibility_screening", "seed": 42}).json()
result = requests.post(f"{BASE}/step", json={"action": {
    "decision": "enroll",
    "rationale": "All criteria met.",
    "regulatory_citation": "Protocol Section 4.1",
    "urgency": "routine",
    "notify_irb": False,
    "notify_sponsor": False
}}).json()
print(result["reward"])
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| API_BASE_URL | LLM endpoint |
| MODEL_NAME | Model name |
| HF_TOKEN | Your API key |

---

All patient data and clinical scenarios are entirely fictional.

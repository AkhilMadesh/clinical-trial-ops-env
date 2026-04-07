"""
ClinicalTrialOps-Env — FastAPI server
Endpoints: GET /reset  POST /step  GET /state  GET /tasks  GET /health
"""
from __future__ import annotations
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os

from env.environment import ClinicalTrialEnv
from env.models import Action

app = FastAPI(
    title="ClinicalTrialOps-Env",
    description=(
        "OpenEnv-compliant environment: AI agent as clinical trial coordinator. "
        "Phase II NSCLC study (LUMINOS-201). Tasks: eligibility screening, "
        "lab dose management, SAE triage."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static landing page
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

_env: Optional[ClinicalTrialEnv] = None


class ResetRequest(BaseModel):
    task_id: str = "eligibility_screening"
    seed: int = 42


class StepRequest(BaseModel):
    action: Action


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"name": "ClinicalTrialOps-Env", "version": "1.0.0",
            "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "env": "ClinicalTrialOps-Env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "eligibility_screening",
                "difficulty": "easy",
                "description": "Enroll or screen-fail 10 patients against protocol criteria.",
                "max_steps": 10,
                "expected_baseline_score": 0.86,
            },
            {
                "id": "lab_dose_management",
                "difficulty": "medium",
                "description": "Manage 12 lab deviation cases with dose modification decisions.",
                "max_steps": 12,
                "expected_baseline_score": 0.59,
            },
            {
                "id": "sae_triage",
                "difficulty": "hard",
                "description": "Triage 15 adverse event reports for SAE classification and FDA reporting.",
                "max_steps": 15,
                "expected_baseline_score": 0.36,
            },
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env
    _env = ClinicalTrialEnv(task_id=req.task_id, seed=req.seed)
    obs = _env.reset()
    return {"observation": obs.model_dump(), "state": _env.state()}


@app.post("/step")
def step(req: StepRequest):
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if _env.state()["done"]:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset.")
    try:
        next_obs, reward, done, info = _env.step(req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "observation": next_obs.model_dump() if next_obs else None,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env.state()


@app.get("/episode_summary")
def episode_summary():
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env.episode_summary()


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False)

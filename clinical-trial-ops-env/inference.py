"""
ClinicalTrialOps-Env — Baseline Inference Script
Uses OpenAI client (compatible with HF Inference API via API_BASE_URL).
Reads credentials from environment variables:
  API_BASE_URL  - LLM endpoint
  MODEL_NAME    - model identifier
  HF_TOKEN      - API key

Run:
  python inference.py

Expected baseline scores (gpt-4o-mini / similar):
  eligibility_screening : ~0.86
  lab_dose_management   : ~0.59
  sae_triage            : ~0.36
"""
from __future__ import annotations
import os
import json
import time
import traceback
import re

from openai import OpenAI
from env.environment import ClinicalTrialEnv
from env.models import Action, Observation
from env.protocol_db import PROTOCOL

# ── Credentials ────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

PROTOCOL_JSON = json.dumps(PROTOCOL, indent=2)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are an experienced clinical trial coordinator for the LUMINOS-201 Phase II NSCLC study.

Here is the exact study protocol knowledge base:
{PROTOCOL_JSON}

Your job is to process patient cases and make protocol-compliant decisions.
First, think step-by-step. Compare the patient's data directly against the relevant protocol criteria (e.g. eligibility, safe lab limits, SAE rules).
Put this step-by-step reasoning inside the "rationale" field of your response.

ALWAYS respond with ONLY a JSON object — no markdown, no preamble, no explanation outside the JSON:
{{
  "decision": "<one of the allowed decisions>",
  "rationale": "<your step-by-step clinical reasoning>",
  "regulatory_citation": "<e.g. 'Protocol Section 6.3' or 'FDA 21 CFR 312.32(c)' or null>",
  "urgency": "<routine | 24h | immediate>",
  "follow_up_action": "<string or null>",
  "notify_irb": false,
  "notify_sponsor": false
}}

Allowed decisions:
  enroll, screen_fail, hold_pending_review,
  dose_reduce_1_level, dose_reduce_2_levels, dose_hold_temporary, dose_discontinue,
  report_SAE, report_non_SAE_AE,
  protocol_deviation_minor, protocol_deviation_major,
  escalate_to_PI, continue_no_action
"""


def obs_to_prompt(obs: Observation) -> str:
    p = obs.patient_data
    labs = {k: v for k, v in p.lab_values.model_dump().items() if v is not None}
    lines = [
        f"CASE TYPE: {obs.case_type.upper().replace('_', ' ')}",
        f"Patient ID: {obs.patient_id}",
        f"Age/Sex: {p.age}{p.sex}  |  Diagnosis: {p.diagnosis}",
        f"ECOG PS: {p.ecog_ps}  |  Prior chemo lines: {p.prior_chemo_lines}",
        f"Medications: {', '.join(p.medications) if p.medications else 'none'}",
        f"Lab values: {json.dumps(labs)}",
    ]
    if obs.case_notes:
        lines.append(f"\nCase notes: {obs.case_notes}")
    if obs.event_description and obs.event_description != obs.case_notes:
        lines.append(f"Event description: {obs.event_description}")
    lines.append(f"\nStep {obs.step_number + 1}, cases remaining: {obs.shift_cases_remaining}")
    return "\n".join(lines)


def parse_action(response_text: str) -> Action:
    """Parse LLM JSON response into Action. Falls back gracefully."""
    text = response_text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        return Action(**data)
    except Exception as e:
        # Fallback: safe default
        return Action(
            decision="hold_pending_review",
            rationale=f"Parse error ({str(e)}) — defaulting to hold pending review.",
            urgency="routine",
        )


def run_task(task_id: str, seed: int = 42, verbose: bool = True) -> float:
    env = ClinicalTrialEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    step_rewards: list[float] = []
    done = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}  |  Model: {MODEL_NAME}")
        print(f"{'='*60}")

    while not done:
        prompt = obs_to_prompt(obs)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=512,
                temperature=0.0,  # deterministic
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            reply = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [API error] {e}")
            reply = '{"decision":"hold_pending_review","rationale":"API error","urgency":"routine"}'

        action = parse_action(reply)
        obs, reward, done, info = env.step(action)
        step_rewards.append(reward)

        if verbose:
            print(
                f"  Step {info['step']+1:>2} | {info['patient_id']} "
                f"| decision={action.decision:<28} "
                f"| reward={reward:+.3f} "
                f"| cumulative={info['cumulative_reward']:.3f}"
            )

        # Small delay to respect rate limits
        time.sleep(0.3)

    mean_score = round(sum(step_rewards) / len(step_rewards), 4)
    summary = env.episode_summary()

    if verbose:
        print(f"\n  Final score : {mean_score}")
        print(f"  Violations  : {summary.get('regulatory_violations', 0)}")
        print(f"  Missed SAEs : {summary.get('missed_critical_events', 0)}")

    return mean_score


def main():
    tasks = [
        "eligibility_screening",
        "lab_dose_management",
        "sae_triage",
    ]
    scores: dict[str, float] = {}
    start = time.time()

    for task_id in tasks:
        try:
            score = run_task(task_id, seed=42, verbose=True)
            scores[task_id] = score
        except Exception:
            print(f"\n[ERROR] Task {task_id} failed:")
            traceback.print_exc()
            scores[task_id] = -1.0

    elapsed = round(time.time() - start, 1)

    print(f"\n{'='*60}")
    print("BASELINE SCORES")
    print(f"{'='*60}")
    for task, score in scores.items():
        bar = "#" * int(score * 30) if score >= 0 else ""
        print(f"  {task:<30} {score:.4f}  {bar}")
    print(f"\nTotal runtime: {elapsed}s")
    print(f"{'='*60}")

    # Machine-readable output
    print(json.dumps({"scores": scores, "model": MODEL_NAME,
                      "seed": 42, "runtime_seconds": elapsed}))


if __name__ == "__main__":
    main()

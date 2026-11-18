"""
api.py -- the triage web service.

Start it:   python -m uvicorn api:app --port 8000
Docs page:  http://127.0.0.1:8000/docs   (FastAPI builds it for you)

POST /predict   counts about recent runs + how long the job ran  ->  P(flaky) and a decision
GET  /health    is the model loaded?  GET /model  the model card
"""
import json
import os
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

import triage_model  # noqa: F401  (the pickled calibrator needs this module importable)
from triage_model import FEATURES, decision_threshold

MODEL_PATH = os.environ.get("TRIAGE_MODEL", "model/triage_model.joblib")
CARD_PATH = os.path.join(os.path.dirname(MODEL_PATH) or ".", "model_card.json")
STATE = {}


@asynccontextmanager
async def lifespan(app):
    t0 = time.perf_counter()
    STATE["bundle"] = joblib.load(MODEL_PATH)
    STATE["card"] = json.load(open(CARD_PATH)) if os.path.exists(CARD_PATH) else {}
    STATE["load_seconds"] = time.perf_counter() - t0
    yield


app = FastAPI(title="CI failure triage", version="0.1", lifespan=lifespan)


class PredictRequest(BaseModel):
    repo_n_1h: int = Field(ge=0, description="runs finished in the project in the last hour")
    repo_fail_1h: int = Field(ge=0, description="... of which failed")
    repo_n_24h: int = Field(ge=0)
    repo_fail_24h: int = Field(ge=0)
    wf_n_7d: int = Field(ge=0, description="runs of THIS workflow in the last 7 days")
    wf_fail_7d: int = Field(ge=0)
    n_failed_jobs_in_run: int = Field(ge=1, description="failed jobs in the same run")
    job_duration_s: float = Field(ge=0, description="how long the failed job ran, in seconds")
    X: float = Field(default=15.0, gt=0, description="cost of a human step, in machine-minutes")
    W: float = Field(default=1.0, ge=0, description="weight of the developer's waiting")

    @model_validator(mode="after")
    def failures_not_more_than_runs(self):
        for a, b in (("repo_fail_1h", "repo_n_1h"), ("repo_fail_24h", "repo_n_24h"),
                     ("wf_fail_7d", "wf_n_7d")):
            if getattr(self, a) > getattr(self, b):
                raise ValueError(f"{a} cannot be larger than {b}")
        return self


class PredictResponse(BaseModel):
    p_flaky: float
    decision: str
    threshold_p: float
    model_trained_at: str | None
    compute_ms: float


@app.get("/health")
def health():
    ok = "bundle" in STATE
    return {"status": "ok" if ok else "loading", "model_load_seconds": STATE.get("load_seconds")}


@app.get("/model")
def model_card():
    return STATE.get("card", {})


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if "bundle" not in STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    t0 = time.perf_counter()
    row = req.model_dump()
    row["repo_fail_rate_24h"] = row["repo_fail_24h"] / max(row["repo_n_24h"], 1)   # same as training
    row["wf_fail_rate_7d"] = row["wf_fail_7d"] / max(row["wf_n_7d"], 1)
    x = pd.DataFrame([row], columns=FEATURES)
    b = STATE["bundle"]
    raw = float(b["pipeline"].predict_proba(x)[0, 1])
    p = float(b["calibrator"].predict(np.array([raw]))[0])
    thr = decision_threshold(req.job_duration_s / 60.0, req.X, req.W)
    return PredictResponse(p_flaky=round(p, 4), decision="rerun" if p > thr else "human",
                           threshold_p=round(thr, 4),
                           model_trained_at=STATE["card"].get("trained_at"),
                           compute_ms=round((time.perf_counter() - t0) * 1000, 3))

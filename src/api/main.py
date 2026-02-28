"""
FastAPI serving layer for The Sentinel MLOps System.

Exposes /health, /predict, /predict/batch, and /model/info endpoints.
Tracks request count, anomaly count, and rolling latency in-process.
"""

import json
import logging
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="The Sentinel — Anomaly Detection API",
    description="Production-ready anomaly detection for machine sensor data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("models/isolation_forest.joblib")
SCALER_PATH = Path("models/scaler.joblib")
META_PATH = Path("models/model_meta.json")

_model = None
_scaler = None
_meta: dict = {}
_request_count: int = 0
_anomaly_count: int = 0
_latencies: list = []


def load_artifacts() -> None:
    global _model, _scaler, _meta
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        logger.info("Isolation Forest loaded.")
    if SCALER_PATH.exists():
        _scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded.")
    if META_PATH.exists():
        _meta = json.loads(META_PATH.read_text())
        logger.info("Model metadata loaded.")


@app.on_event("startup")
def startup() -> None:
    load_artifacts()


# ---------- Schemas ----------

class PredictRequest(BaseModel):
    value: float
    value_rolling_mean: float = 0.0
    value_rolling_std: float = 0.0
    value_diff: float = 0.0


class PredictResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    latency_ms: float


class BatchPredictRequest(BaseModel):
    instances: List[PredictRequest]


# ---------- Endpoints ----------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "request_count": _request_count,
        "anomaly_count": _anomaly_count,
        "avg_latency_ms": round(
            float(np.mean(_latencies[-100:])) if _latencies else 0.0, 2
        ),
    }


@app.get("/model/info")
def model_info() -> dict:
    return _meta


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    global _request_count, _anomaly_count

    if _model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()
    X = np.array(
        [[req.value, req.value_rolling_mean, req.value_rolling_std, req.value_diff]]
    )
    X_scaled = _scaler.transform(X)
    pred = int(_model.predict(X_scaled)[0])
    score = float(_model.score_samples(X_scaled)[0])
    latency_ms = (time.perf_counter() - t0) * 1000

    is_anomaly = pred == -1
    _request_count += 1
    if is_anomaly:
        _anomaly_count += 1
    _latencies.append(latency_ms)

    return PredictResponse(
        is_anomaly=is_anomaly,
        anomaly_score=score,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest) -> dict:
    global _request_count, _anomaly_count

    if _model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()
    X = np.array(
        [
            [r.value, r.value_rolling_mean, r.value_rolling_std, r.value_diff]
            for r in req.instances
        ]
    )
    X_scaled = _scaler.transform(X)
    preds = _model.predict(X_scaled)
    scores = _model.score_samples(X_scaled).tolist()
    latency_ms = (time.perf_counter() - t0) * 1000

    results = []
    for pred, score in zip(preds, scores):
        is_anomaly = int(pred) == -1
        _request_count += 1
        if is_anomaly:
            _anomaly_count += 1
        results.append({"is_anomaly": is_anomaly, "anomaly_score": score})

    _latencies.append(latency_ms)
    return {"predictions": results, "latency_ms": round(latency_ms, 2)}

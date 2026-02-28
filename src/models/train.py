"""
Model training script for The Sentinel MLOps System.

Trains an Isolation Forest anomaly detector on machine temperature sensor data
and logs every experiment run to MLflow for full reproducibility.
"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = Path("data/processed/machine_temperature_processed.csv")
MODEL_DIR = Path("models")
MLFLOW_EXPERIMENT = "sentinel-anomaly-detection"
FEATURE_COLS = ["value", "value_rolling_mean", "value_rolling_std", "value_diff"]


def load_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df)} records from {path}")
    return df


def train(
    n_estimators: int = 100,
    contamination: float = 0.05,
    max_samples: str = "auto",
    random_state: int = 42,
) -> dict:
    """Train Isolation Forest, log to MLflow, and persist artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_data()
    X = df[FEATURE_COLS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "contamination": contamination,
                "max_samples": max_samples,
                "random_state": random_state,
                "features": str(FEATURE_COLS),
                "n_samples": len(X),
            }
        )

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
        )
        model.fit(X_scaled)

        preds = model.predict(X_scaled)
        scores = model.score_samples(X_scaled)

        n_anomalies = int((preds == -1).sum())
        anomaly_rate = round(n_anomalies / len(preds), 4)
        avg_score = float(np.mean(scores))

        mlflow.log_metrics(
            {
                "n_anomalies": n_anomalies,
                "anomaly_rate": anomaly_rate,
                "avg_anomaly_score": avg_score,
                "n_samples": len(X),
            }
        )

        model_path = MODEL_DIR / "isolation_forest.joblib"
        scaler_path = MODEL_DIR / "scaler.joblib"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))

        meta = {
            "run_id": run_id,
            "n_estimators": n_estimators,
            "contamination": contamination,
            "max_samples": max_samples,
            "random_state": random_state,
            "n_anomalies": n_anomalies,
            "anomaly_rate": anomaly_rate,
            "avg_anomaly_score": avg_score,
            "features": FEATURE_COLS,
            "n_samples": len(X),
        }
        meta_path = MODEL_DIR / "model_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(meta_path))

        logger.info(
            f"Training complete. Anomalies: {n_anomalies} / {len(X)} ({anomaly_rate:.2%})"
        )
        return meta


if __name__ == "__main__":
    train()

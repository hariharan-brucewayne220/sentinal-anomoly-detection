"""
Drift monitoring module for The Sentinel MLOps System.

Uses Evidently AI to generate data drift and model performance reports
by comparing a reference (training) dataset against a current (production) window.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = Path("data/processed/machine_temperature_processed.csv")
REPORTS_DIR = Path("reports")
FEATURE_COLS = ["value", "value_rolling_mean", "value_rolling_std", "value_diff"]


def load_reference_and_current(
    reference_frac: float = 0.6, current_window: int = 500
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the processed dataset into a reference (training) slice
    and a current (production simulation) slice.
    """
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["timestamp"])
    split = int(len(df) * reference_frac)
    reference = df.iloc[:split][FEATURE_COLS].copy()
    current = df.iloc[split : split + current_window][FEATURE_COLS].copy()
    return reference, current


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path = REPORTS_DIR / "drift_report.html",
) -> dict:
    """Generate an Evidently data drift report and return a summary dict."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    column_mapping = ColumnMapping(numerical_features=FEATURE_COLS)
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    report.save_html(str(output_path))
    logger.info(f"Drift report saved to {output_path}")

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"].get("dataset_drift", False)
    drifted_features = result["metrics"][0]["result"].get("number_of_drifted_columns", 0)
    total_features = result["metrics"][0]["result"].get("number_of_columns", len(FEATURE_COLS))

    summary = {
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "total_features": total_features,
        "drift_share": round(drifted_features / max(total_features, 1), 4),
        "report_path": str(output_path),
    }

    summary_path = REPORTS_DIR / "drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Drift summary: {summary}")
    return summary


def run() -> dict:
    """Full drift monitoring pipeline."""
    reference, current = load_reference_and_current()
    return generate_drift_report(reference, current)


if __name__ == "__main__":
    run()

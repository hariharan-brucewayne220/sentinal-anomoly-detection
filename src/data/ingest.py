"""
Data ingestion module for The Sentinel MLOps System.

Pulls machine temperature sensor data from the NAB (Numenta Anomaly Benchmark)
public dataset on GitHub — no API key required.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = (
    "https://raw.githubusercontent.com/numenta/NAB/master/data/"
    "realKnownCause/machine_temperature_system_failure.csv"
)
RAW_DATA_PATH = Path("data/raw/machine_temperature.csv")
PROCESSED_DATA_PATH = Path("data/processed/machine_temperature_processed.csv")
META_PATH = Path("data/raw/meta.json")


def fetch_data(url: str = DATA_URL, output_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Download and save the raw dataset from a public URL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching data from {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    logger.info(f"Raw data saved to {output_path}")

    df = pd.read_csv(output_path)
    logger.info(f"Loaded {len(df)} records, columns: {list(df.columns)}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps, sort, drop nulls, and engineer rolling features."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    window = 12
    df["value_rolling_mean"] = df["value"].rolling(window=window, min_periods=1).mean()
    df["value_rolling_std"] = (
        df["value"].rolling(window=window, min_periods=1).std().fillna(0)
    )
    df["value_diff"] = df["value"].diff().fillna(0)
    return df


def run() -> pd.DataFrame:
    """Full ingestion pipeline: fetch → preprocess → save."""
    df = fetch_data()
    df = preprocess(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Processed data ({len(df)} rows) saved to {PROCESSED_DATA_PATH}")

    meta = {
        "ingested_at": datetime.utcnow().isoformat(),
        "records": len(df),
        "columns": list(df.columns),
        "source_url": DATA_URL,
        "value_min": float(df["value"].min()),
        "value_max": float(df["value"].max()),
        "value_mean": float(df["value"].mean()),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info(f"Metadata saved to {META_PATH}")
    return df


if __name__ == "__main__":
    run()

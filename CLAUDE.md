# The Sentinel MLOps System — CLAUDE.md

## What This Project Is
A production-ready anomaly detection system that monitors machine temperature sensor data in real time. It represents a full MLOps lifecycle: data ingestion → model training → API serving → drift monitoring → dashboard → CI/CD.

**GitHub:** https://github.com/hariharan-brucewayne220/sential-anomoly-detection

---

## Project Structure
```
newproject/
├── src/
│   ├── data/ingest.py          # Pulls NAB CSV, engineers rolling features
│   ├── models/train.py         # Trains Isolation Forest, logs to MLflow
│   ├── api/main.py             # FastAPI: /health /predict /predict/batch /model/info
│   ├── monitoring/drift.py     # Evidently AI drift report generation
│   └── ui/app.py               # 4-screen Streamlit dashboard (dark theme)
├── tests/
│   ├── test_api.py             # 6 API tests
│   └── test_model.py           # 5 model/preprocessing tests
├── data/
│   ├── raw/                    # machine_temperature.csv + meta.json
│   └── processed/              # machine_temperature_processed.csv
├── models/                     # isolation_forest.joblib, scaler.joblib, model_meta.json
├── reports/                    # drift_report.html, drift_summary.json
├── .github/workflows/ci.yml    # lint → test → docker build → smoke test
├── .streamlit/config.toml      # dark theme config, headless, no usage stats
├── Dockerfile                  # multi-stage build
├── docker-compose.yml          # api + dashboard services
└── requirements.txt
```

---

## Stack
| Layer | Tool |
|-------|------|
| Model | Isolation Forest (scikit-learn) — unsupervised anomaly detection |
| Data | NAB machine temperature CSV from GitHub (22,695 rows, 4 known failures) |
| Features | value, value_rolling_mean, value_rolling_std, value_diff (window=12) |
| Tracking | MLflow (local mlruns/) |
| API | FastAPI + Uvicorn — port 8000 |
| Dashboard | Streamlit — port 8501 |
| Drift | Evidently AI |
| CI/CD | GitHub Actions |
| Container | Docker multi-stage |

---

## Run Commands

> IMPORTANT: Always set `TEMP=d:/tmp` — C: drive is nearly full (only ~17MB free)

```bash
# 1. Ingest data
PYTHONPATH=. TEMP=d:/tmp venv/Scripts/python -m src.data.ingest

# 2. Train model
PYTHONPATH=. TEMP=d:/tmp venv/Scripts/python -m src.models.train

# 3. Run tests (11/11 passing)
PYTHONPATH=. TEMP=d:/tmp venv/Scripts/python -m pytest tests/ -v

# 4. Generate drift report
PYTHONPATH=. TEMP=d:/tmp venv/Scripts/python -m src.monitoring.drift

# 5. Start API (port 8000)
PYTHONPATH=. TEMP=d:/tmp venv/Scripts/uvicorn src.api.main:app --reload --port 8000

# 6. Start dashboard (port 8501)
PYTHONPATH=. TEMP=d:/tmp venv/Scripts/streamlit run src/ui/app.py

# 7. MLflow UI (port 5000)
TEMP=d:/tmp venv/Scripts/mlflow ui --port 5000

# 8. Install packages (pip needs TEMP on D: to avoid C: full error)
TMPDIR=d:/tmp TEMP=d:/tmp TMP=d:/tmp venv/Scripts/pip install -r requirements.txt
```

---

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Status, request count, anomaly count, avg latency |
| GET | /model/info | Hyperparams, run_id, metrics from last training |
| POST | /predict | Single reading → is_anomaly, anomaly_score, latency_ms |
| POST | /predict/batch | List of readings → batch predictions |
| GET | /docs | Auto-generated interactive FastAPI docs |

### Example predict call
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"value": 500.0, "value_rolling_mean": 80.0, "value_rolling_std": 5.0, "value_diff": 420.0}'
# → {"is_anomaly": true, "anomaly_score": -0.71, "latency_ms": 10.83}
```

---

## Model Performance

### Dataset
- 22,695 total sensor readings
- 2,268 ground-truth anomalies (9.99% true anomaly rate)
- 4 known real machine failure windows from NAB labels:
  - 2013-12-10 06:25 → 2013-12-12 05:35
  - 2013-12-15 17:50 → 2013-12-17 17:00
  - 2014-01-27 14:20 → 2014-01-29 13:30
  - 2014-02-07 14:55 → 2014-02-09 14:05

### MLflow Experiment Runs (sentinel-anomaly-detection)

| Run | contamination | Predicted Anomalies | Precision | Recall | F1 | ROC-AUC |
|-----|--------------|---------------------|-----------|--------|----|---------|
| Run 1 | 0.05 | 1,135 (5.00%) | 59.7% | 29.9% | 39.9% | 0.8233 |
| Run 2 | 0.10 | 2,270 (10.00%) | 48.2% | 48.3% | **48.3%** | 0.8233 |

**Active model: Run 2 (contamination=0.10)**

### Final Model Metrics (contamination=0.10)

| Metric | Score | Notes |
|--------|-------|-------|
| ROC-AUC | **0.8233** | Strong ranking — model scoring is well-calibrated |
| Precision | 48.2% | ~1 in 2 alerts is a real anomaly |
| Recall | 48.3% | Catches ~half of all true failures |
| F1 Score | 48.3% | Balanced precision/recall |

### Confusion Matrix (Run 2)
```
                  Predicted Normal   Predicted Anomaly
Actual Normal          19,252              1,175   (False Positives)
Actual Anomaly          1,173              1,095   (True Positives)
```
- **True Positives:** 1,095 — real failures correctly caught
- **False Positives:** 1,175 — false alarms (engineer checks, machine is fine)
- **False Negatives:** 1,173 — missed failures (machine breaks undetected)
- **True Negatives:** 19,252 — correctly identified as normal

### Why contamination=0.10 is better than 0.05
- True anomaly rate in dataset is **9.99%** — so 0.10 matches reality
- At 0.05: precision=60% but recall=30% — misses 70% of failures (too dangerous)
- At 0.10: precision=recall=48% — balanced, catches more failures
- ROC-AUC stays identical (0.8233) — the model's internal scoring didn't change, only the decision threshold

### Why Recall > Precision for this use case
- **Missing a failure** = machine breaks, production halts, costly repair
- **False alarm** = engineer checks machine for 5 minutes, finds nothing
- Cost of missing a failure >> cost of a false alarm → optimize for recall

---

## Known Issues / Workarounds
- **C: drive nearly full** — always set `TEMP=d:/tmp` before pip, MLflow, or uvicorn
- `mlflow.sklearn.log_model` was removed from train.py — it writes to C:\temp and crashes. Joblib artifacts are logged instead
- Streamlit needs `--server.headless true` and `--browser.gatherUsageStats false` flags, or use `.streamlit/config.toml` (already present)
- venv uses Python 3.10.0 (system Python) even though requirements say 3.11 — works fine in practice
- IDE shows import errors for numpy/pandas/plotly in app.py — false positives, linter doesn't see the venv. Code runs correctly

---

## Dashboard Screens
1. **System Overview** — live sensor chart with anomaly markers, pie chart, feature drift bars, alert timeline
2. **Model Registry** — active model details, experiment history table, gauge chart
3. **CI/CD & API Health** — pipeline step status, latency chart, live API test widget
4. **Data Pipeline** — ingestion status, dataset preview, DVC version history, Evidently report

---

## What's Not Done Yet (PRD gaps)
- DVC not configured — `dvc init` and remote storage (DagsHub/Google Drive) not set up
- Not deployed publicly — only runs locally. Deployment target: Hugging Face Spaces (Streamlit) + Railway/Render (FastAPI)
- No README.md written yet

---

## Git
```bash
cd d:/claude-projects/newproject
git add .
git commit -m "your message"
git push origin main
```
Remote: https://github.com/hariharan-brucewayne220/sential-anomoly-detection.git

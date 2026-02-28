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
Evaluated against NAB ground-truth labels (4 known machine failure windows):

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.8233 |
| Precision | 48.2% |
| Recall | 48.3% |
| F1 Score | 48.3% |
| contamination | 0.10 (matches true 9.99% anomaly rate) |

Two MLflow runs logged:
- Run 1: contamination=0.05 → F1=0.40 (too few anomalies flagged)
- Run 2: contamination=0.10 → F1=0.48 (matches true rate, current active model)

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

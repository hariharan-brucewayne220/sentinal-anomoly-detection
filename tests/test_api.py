"""Unit tests for the Sentinel FastAPI serving layer."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture()
def client():
    """Create a test client with mocked model artifacts."""
    with patch("src.api.main.load_artifacts"):
        from src.api import main as api_module

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])       # normal
        mock_model.score_samples.return_value = np.array([-0.1])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.0, 0.0, 0.0, 0.0]])

        api_module._model = mock_model
        api_module._scaler = mock_scaler
        api_module._request_count = 0
        api_module._anomaly_count = 0
        api_module._latencies = []

        with TestClient(api_module.app) as c:
            yield c


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_normal(client):
    resp = client.post(
        "/predict",
        json={
            "value": 80.0,
            "value_rolling_mean": 79.5,
            "value_rolling_std": 1.2,
            "value_diff": 0.5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_anomaly"] is False
    assert "anomaly_score" in body
    assert body["latency_ms"] >= 0


def test_predict_anomaly(client):
    from src.api import main as api_module

    api_module._model.predict.return_value = np.array([-1])  # anomaly

    resp = client.post(
        "/predict",
        json={
            "value": 500.0,
            "value_rolling_mean": 80.0,
            "value_rolling_std": 5.0,
            "value_diff": 420.0,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["is_anomaly"] is True


def test_predict_increments_request_count(client):
    from src.api import main as api_module

    before = api_module._request_count
    client.post(
        "/predict",
        json={"value": 80.0, "value_rolling_mean": 79.5, "value_rolling_std": 1.2, "value_diff": 0.5},
    )
    assert api_module._request_count == before + 1


def test_predict_batch(client):
    from src.api import main as api_module

    api_module._model.predict.return_value = np.array([1, -1])
    api_module._model.score_samples.return_value = np.array([-0.1, -0.5])
    api_module._scaler.transform.return_value = np.zeros((2, 4))

    resp = client.post(
        "/predict/batch",
        json={
            "instances": [
                {"value": 80.0, "value_rolling_mean": 79.5, "value_rolling_std": 1.2, "value_diff": 0.5},
                {"value": 500.0, "value_rolling_mean": 80.0, "value_rolling_std": 5.0, "value_diff": 420.0},
            ]
        },
    )
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["is_anomaly"] is False
    assert preds[1]["is_anomaly"] is True


def test_predict_no_model_returns_503():
    """When no model is loaded, /predict should return 503."""
    with patch("src.api.main.load_artifacts"):
        from src.api import main as api_module

        api_module._model = None
        api_module._scaler = None

        with TestClient(api_module.app) as c:
            resp = c.post(
                "/predict",
                json={"value": 80.0, "value_rolling_mean": 79.0, "value_rolling_std": 1.0, "value_diff": 1.0},
            )
            assert resp.status_code == 503

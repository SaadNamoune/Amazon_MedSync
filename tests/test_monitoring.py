import math
import sys
from pathlib import Path

import mlflow
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medsync.api.main import app  # noqa: E402
from medsync.api.monitoring import _json_safe  # noqa: E402

client = TestClient(app)


def test_json_safe_converts_nan_and_inf_to_none():
    assert _json_safe(float("nan")) is None
    assert _json_safe(float("inf")) is None
    assert _json_safe(0.5) == 0.5
    assert _json_safe(None) is None


def test_training_history_handles_nan_metrics_without_500(tmp_path):
    # Regression test: a run with a NaN metric (e.g. macro AUC on an eval
    # set too small for any label to have both classes present) used to
    # crash this endpoint with a 500, since Starlette's JSONResponse
    # rejects NaN as invalid JSON.
    original_uri = mlflow.get_tracking_uri()
    try:
        tracking_dir = tmp_path / "mlruns"
        mlflow.set_tracking_uri(tracking_dir.as_uri())
        mlflow.set_experiment("test-nan-experiment")
        with mlflow.start_run():
            mlflow.log_params({"num_nodes": 5, "target_epsilon": 1.0})
            mlflow.log_metrics({
                "macro_auc": float("nan"),
                "avg_client_loss": 0.5,
                "max_client_epsilon": 0.8,
            }, step=1)

        resp = client.get("/api/training-history", params={"experiment": "test-nan-experiment"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["rounds"][0]["macro_auc"] is None
        assert body["rounds"][0]["avg_client_loss"] == 0.5
    finally:
        mlflow.set_tracking_uri(original_uri)


def test_monitoring_page_serves_html():
    resp = client.get("/monitoring")
    assert resp.status_code == 200
    assert "Federation Monitor" in resp.text


def test_training_history_unknown_experiment_404s():
    resp = client.get("/api/training-history", params={"experiment": "does-not-exist"})
    assert resp.status_code == 404


def test_nodes_returns_list():
    resp = client.get("/api/nodes")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_experiments_returns_list():
    resp = client.get("/api/experiments")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

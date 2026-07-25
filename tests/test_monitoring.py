import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medsync.api.main import app  # noqa: E402

client = TestClient(app)


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

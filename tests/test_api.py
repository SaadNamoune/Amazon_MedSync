import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medsync.api.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_dashboard_serves_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "MedSync Diagnostic Dashboard" in resp.text


def _fake_jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(100, 100, 100)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_predict_returns_all_labels():
    from medsync.data.dataset import LABEL_NAMES

    resp = client.post(
        "/predict",
        files={"file": ("xray.jpg", _fake_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["findings"].keys()) == set(LABEL_NAMES)
    assert all(0.0 <= p <= 1.0 for p in body["findings"].values())
    assert body["inference_ms"] > 0


def test_predict_rejects_bad_content_type():
    resp = client.post(
        "/predict",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 400

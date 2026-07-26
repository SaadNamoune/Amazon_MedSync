import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medsync.api.main import app  # noqa: E402
from conftest import TEST_USERNAME, TEST_PASSWORD  # noqa: E402

client = TestClient(app)


def _login(username=TEST_USERNAME, password=TEST_PASSWORD):
    return client.post("/auth/login", data={"username": username, "password": password})


def test_login_succeeds_with_correct_credentials():
    resp = _login()
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_fails_with_wrong_password():
    resp = _login(password="wrong")
    assert resp.status_code == 401


def test_predict_requires_auth():
    resp = client.post("/predict", files={"file": ("x.jpg", io.BytesIO(b"x"), "image/jpeg")})
    assert resp.status_code == 401


def test_predict_works_with_valid_token(auth_token):
    buf = io.BytesIO()
    Image.new("RGB", (224, 224)).save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {auth_token}"},
        files={"file": ("xray.jpg", buf, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert "findings" in resp.json()


def test_login_page_serves_html():
    resp = client.get("/login")
    assert resp.status_code == 200
    assert "Sign in" in resp.text

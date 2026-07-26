import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medsync.api.main import app  # noqa: E402
from medsync.auth import hash_password  # noqa: E402
from medsync.db import Base, User, get_db  # noqa: E402

TEST_USERNAME = "testuser"
TEST_PASSWORD = "testpass123"

_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # one shared connection, or each session gets its own blank :memory: db
)
_TestSession = sessionmaker(bind=_engine)
Base.metadata.create_all(_engine)


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db

_seed_db = _TestSession()
if not _seed_db.query(User).filter(User.username == TEST_USERNAME).first():
    _seed_db.add(User(username=TEST_USERNAME, hashed_password=hash_password(TEST_PASSWORD), role="clinician"))
    _seed_db.commit()
_seed_db.close()


@pytest.fixture
def auth_token(request):
    """A valid bearer token for TEST_USERNAME, for tests that need to call
    protected endpoints like /predict."""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.post("/auth/login", data={"username": TEST_USERNAME, "password": TEST_PASSWORD})
    return resp.json()["access_token"]

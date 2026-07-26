"""
Diagnostic dashboard API: serves the current global federated model and
answers inference queries. This is what a hospital's clinicians would hit
after pulling the latest aggregated model -- it never sees other hospitals'
training data, only the resulting weights.
"""
import io
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from medsync.models.chexnet import build_chexnet  # noqa: E402
from medsync.data.dataset import LABEL_NAMES, build_transform  # noqa: E402
from medsync.api.monitoring import router as monitoring_router  # noqa: E402
from medsync.auth import authenticate_user, create_access_token, get_current_user  # noqa: E402
from medsync.db import PredictionLog, User, get_db, init_db  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
MODEL_PATH = Path("global_model_final.pth")
_device = "cuda" if torch.cuda.is_available() else "cpu"
_transform = build_transform(train=False)
_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="MedSync Diagnostic API", version="0.1.0", lifespan=lifespan)
app.include_router(monitoring_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/login")
def login_page():
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/monitoring")
def monitoring_page():
    return FileResponse(STATIC_DIR / "monitoring.html")


@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(401, "Incorrect username or password")
    return {
        "access_token": create_access_token(user.username),
        "token_type": "bearer",
        "role": user.role,
    }


def get_model():
    global _model
    if _model is None:
        _model = build_chexnet(pretrained=MODEL_PATH.exists() is False)
        if MODEL_PATH.exists():
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        _model.to(_device).eval()
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists(), "device": _device}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "Upload a JPEG or PNG chest X-ray image")

    start = time.perf_counter()
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(_device)

    model = get_model()
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze(0).cpu().tolist()

    elapsed_ms = (time.perf_counter() - start) * 1000
    findings = {name: round(prob, 4) for name, prob in zip(LABEL_NAMES, probs)}
    sorted_findings = dict(sorted(findings.items(), key=lambda kv: -kv[1]))
    top_finding, top_probability = next(iter(sorted_findings.items()))

    db.add(PredictionLog(
        user_id=current_user.id,
        top_finding=top_finding,
        top_probability=top_probability,
        inference_ms=elapsed_ms,
    ))
    db.commit()

    return {
        "findings": sorted_findings,
        "inference_ms": round(elapsed_ms, 1),
    }

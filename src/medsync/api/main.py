"""
Diagnostic dashboard API: serves the current global federated model and
answers inference queries. This is what a hospital's clinicians would hit
after pulling the latest aggregated model -- it never sees other hospitals'
training data, only the resulting weights.
"""
import io
import sys
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from medsync.models.chexnet import build_chexnet  # noqa: E402
from medsync.data.dataset import LABEL_NAMES, build_transform  # noqa: E402

app = FastAPI(title="MedSync Diagnostic API", version="0.1.0")

MODEL_PATH = Path("global_model_final.pth")
_device = "cuda" if torch.cuda.is_available() else "cpu"
_transform = build_transform(train=False)
_model = None


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
async def predict(file: UploadFile = File(...)):
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
    return {
        "findings": dict(sorted(findings.items(), key=lambda kv: -kv[1])),
        "inference_ms": round(elapsed_ms, 1),
    }

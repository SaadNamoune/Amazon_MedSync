# MedSync — Federated AI Diagnostic Network

A federated learning platform that lets hospitals collaboratively train a
chest X-ray diagnostic model without any raw patient data leaving their
premises. Each simulated hospital node trains locally with differentially
private SGD (Opacus); only noised model weights are sent to the server,
which aggregates them with FedAvg.

## Status

This is an early working prototype, not the final target system. It
validates the full pipeline end-to-end on a small (~3k image) subset of
NIH ChestX-ray14, not the full 112k-image dataset the proposal's AUC ≥ 0.85
target assumes. Scaling to the full dataset and multi-day training is the
next milestone — see "Known limitations" below.

## Stack

- **Model**: DenseNet121 / CheXNet architecture (torchvision, ImageNet-pretrained)
- **Federation**: custom FedAvg simulator (`src/medsync/federation/`) — see
  "Alternatives considered"
- **Privacy**: Opacus DP-SGD per client, per-round epsilon tracked and logged
- **Tracking**: MLflow
- **Serving**: FastAPI diagnostic API
- **Deployment**: Docker Compose (hospital node + MLflow server)

## Alternatives considered

NVIDIA FLARE was the original plan (per the project proposal) for
production-grade federated orchestration. It's installed
(`requirements.txt`) but the training loop here is a custom, from-scratch
FedAvg simulator instead of a FLARE Job. Reasoning: FLARE's Job API adds
real setup and debugging surface (client/server process management, job
config, secure provisioning) that isn't needed to validate the core
ML/privacy pipeline first. Once the modeling side is proven out, migrating
the same client/server/aggregation logic into a FLARE Job is the natural
next step for real multi-machine deployment (cross-institution networking,
TLS, admin console) — the current code is structured so that migration is
mostly plumbing, not a rewrite.

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Requires an NVIDIA GPU (tested on RTX 3060, 6GB VRAM) with CUDA 12.1 drivers.

## Running the pipeline

```powershell
# 1. Pull a subset of NIH ChestX-ray14 from Hugging Face (no auth needed)
python scripts/download_data.py --num-images 3000

# 2. Split into 5 non-IID simulated hospital nodes
python scripts/partition_data.py --num-nodes 5 --alpha 0.5

# 3. Run federated training (FedAvg + DP-SGD), logs to MLflow
python scripts/run_federation.py --rounds 5 --local-epochs 1

# 4. Inspect metrics
mlflow ui   # http://localhost:5000

# 5. Serve the resulting global model
uvicorn medsync.api.main:app --reload --app-dir src
```

## Docker

```powershell
docker compose up --build
```

Brings up an MLflow tracking server and one hospital-node diagnostic API
(`POST /predict` with a chest X-ray image, `GET /health`).

## Known limitations / next steps

- **Dataset size**: prototyped on ~3,000 images (600/node) for iteration
  speed, not the full 112,120-image NIH set. AUC numbers from this subset
  are a pipeline sanity check, not the final accuracy target — do not
  quote them as the project's headline metric.
- **Single machine**: all 5 "hospital nodes" are simulated in one process
  on one GPU. Real deployment needs one node per physical/cloud location.
- **NVFlare migration**: see "Alternatives considered" above.
- **Privacy accounting**: per-round epsilon is tracked per client via
  Opacus's accountant; a cross-round composition analysis (total epsilon
  spent over all rounds, not just per-round) is not yet implemented.

# MedSync — Federated AI Diagnostic Network

A federated learning platform that lets hospitals collaboratively train a
chest X-ray diagnostic model without any raw patient data leaving their
premises. Each simulated hospital node trains locally with differentially
private SGD (Opacus); only noised model weights are sent to the server,
which aggregates them with FedAvg.

## Status

This is an early working prototype, not the final target system. It
validates the full pipeline end-to-end on a small (600 image) subset of
NIH ChestX-ray14, not the full 112k-image dataset the proposal's AUC ≥ 0.85
target assumes. Scaling to the full dataset and multi-day training is the
next milestone — see "Known limitations" below.

### First real training run (5 rounds, 600 images / 5 non-IID nodes)

```
node sizes (train/held-out): node_0=147/25 node_1=63/10 node_2=51/8 node_3=181/31 node_4=72/12

round  avg_client_loss  global_macro_auc  max_client_epsilon
1      0.7006           0.4538            0.962
2      0.6845           0.4470            0.962
3      0.6704           0.4332            0.962
4      0.6559           0.4694            0.962
5      0.6425           0.4660            0.962
```

Read honestly: local training loss falls every round (the DP-SGD + FedAvg
mechanics work correctly end-to-end, on GPU, with a real dataset), and the
per-round privacy budget stays under the ε≤1.0 target throughout. But macro
AUC sits at ~0.43–0.51 — essentially chance-level for a 15-way multi-label
problem. That's expected at this scale (as few as 51 training images on the
smallest node, under strong DP noise) and is **not** the project's accuracy
result — it's confirmation the pipeline is wired correctly. Getting a real
AUC signal needs the full dataset (see "Known limitations").

**Follow-up experiment** (same 600 images, `--rounds 20 --local-epochs 3`
instead of `5`/`1`): macro AUC rose to a peak of ~0.586 by round 5, then
oscillated and drifted down to ~0.52–0.55 through round 20, rather than
continuing to improve. That's the signature of a training-duration ceiling,
not a training-duration bottleneck: more rounds on the same 51-181
images/node buys a small one-time gain and then just adds noise. Confirms
the real fix is more data, not more rounds -- see "Known limitations."

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
#    Note: streams from NIH's remote archive at roughly 0.1-0.15 images/sec
#    (~8.6 sec/image measured), so 600 images takes ~85 minutes and scales
#    linearly -- 2500 images is ~6 hours. Re-running with a higher
#    --num-images resumes (skips already-downloaded rows) rather than
#    duplicating them, but skipping still re-walks the remote archive at
#    the same rate, so it does NOT save wall-clock time on this source --
#    it only avoids corrupting/duplicating what's already on disk.
python scripts/download_data.py --num-images 600

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

- **Dataset size**: prototyped on 600 images total (51-181/node after
  non-IID partitioning) for iteration speed, not the full 112,120-image
  NIH set. AUC numbers from this subset are a pipeline sanity check, not
  the final accuracy target — do not quote them as the project's headline
  metric.
- **Single machine**: all 5 "hospital nodes" are simulated in one process
  on one GPU. Real deployment needs one node per physical/cloud location.
- **NVFlare migration**: see "Alternatives considered" above.
- **Privacy accounting**: per-round epsilon is tracked per client via
  Opacus's accountant; a cross-round composition analysis (total epsilon
  spent over all rounds, not just per-round) is not yet implemented.

# MedSync — Federated AI Diagnostic Network

A federated learning platform that lets hospitals collaboratively train a
chest X-ray diagnostic model without any raw patient data leaving their
premises. Each simulated hospital node trains locally with differentially
private SGD (Opacus); only noised model weights are sent to the server,
which aggregates them with FedAvg.

## Status

This is an early working prototype, not the final target system. It
validates the full pipeline end-to-end on real NIH ChestX-ray14 data
(currently the 5,606-image official Kaggle sample, not the full
112,120-image set the proposal's AUC ≥ 0.85 target assumes). Scaling to
the full dataset and multi-day training is the next milestone — see
"Known limitations" below.

Two web interfaces are live: a clinician-facing **diagnostic dashboard**
(`/`, upload an X-ray, get findings) and an ops-facing **federation
monitor** (`/monitoring`, per-round AUC/loss/privacy-budget charts and
per-node status), both served by the same FastAPI app.

### Experiment history (chronological, all real runs -- see `mlruns/`)

**1. Pipeline sanity check** (600 images via HF streaming, 5 rounds x 1
local epoch): macro AUC 0.43–0.51, essentially chance. Confirmed the
DP-SGD + FedAvg mechanics work end-to-end (loss fell every round, ε
stayed under 1.0) but wasn't a real accuracy result -- not enough data.

**2. Longer training, same 600 images** (20 rounds x 3 local epochs):
peaked at macro AUC ~0.586 (round 5), then drifted down to ~0.52–0.55.
Confirmed the bottleneck was data volume, not training duration --
more rounds on the same small shard doesn't help past an early plateau.

**3. Full 5,606-image Kaggle sample, non-IID across 5 nodes** (node
sizes 363-2305 after partitioning; 10 rounds x 2 local epochs):

```
round  avg_client_loss (largest node)  global_macro_auc  max_client_epsilon
1      0.3646                          0.5059            0.867
2      0.1838                          0.5233            0.867
3      0.1471                          0.5368            0.867
5      0.1291                          0.5394            0.867
8      0.1182                          0.5412            0.867
10     0.1158                          0.5575            0.867
```

This is the first genuinely-above-chance result: macro AUC reached
**0.558**, a real (if still modest) signal, with the per-round privacy
budget holding under ε≤0.87 throughout. Still well short of the 0.85
target -- that requires the full 112k-image dataset -- but it's an
honest, reproducible data point rather than a guess.

**Data source note**: the initial plan was to stream from Hugging Face
(`scripts/download_data.py`), but that source's remote archive turned out
to be extremely slow (~8.6 sec/image measured, no way to speed up via
resume/skip). Switched to Kaggle's bulk zip of the same official NIH
sample (`scripts/convert_kaggle_sample.py`) for run #3 -- same data,
downloads in ~2 minutes instead of ~13 hours. Both scripts are kept;
HF streaming still works, it's just the slower option.

## Stack

- **Model**: DenseNet121 / CheXNet architecture (torchvision, ImageNet-pretrained)
- **Federation**: custom FedAvg simulator (`src/medsync/federation/`) — see
  "Alternatives considered"
- **Privacy**: Opacus DP-SGD per client, per-round epsilon tracked and logged
- **Tracking**: MLflow
- **Serving**: FastAPI, two interfaces -- diagnostic dashboard (`/`) and
  federation monitor (`/monitoring`)
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
# 1. Get data -- two options:

#    (a) Kaggle bulk download (recommended, ~2 minutes for 5,606 images).
#        Needs a Kaggle API token at C:\Users\<you>\.kaggle\access_token
#        (from kaggle.com -> Settings -> API -> Create New Token).
kaggle datasets download -d nih-chest-xrays/sample -p <some-dir>
# unzip it, then:
python scripts/convert_kaggle_sample.py --kaggle-dir <extracted-dir>

#    (b) Hugging Face streaming (no auth, but slow: ~8.6 sec/image, so
#        600 images takes ~85 min and it scales linearly -- 2500 would be
#        ~6 hours). Re-running with a higher --num-images resumes (skips
#        already-downloaded rows) rather than duplicating them, but
#        skipping still re-walks the remote archive at the same rate, so
#        it does NOT save wall-clock time on this source.
python scripts/download_data.py --num-images 600

# 2. Split into 5 non-IID simulated hospital nodes
python scripts/partition_data.py --num-nodes 5 --alpha 0.5

# 3. Run federated training (FedAvg + DP-SGD), logs to MLflow
python scripts/run_federation.py --rounds 10 --local-epochs 2

# 4. Inspect metrics
mlflow ui   # http://localhost:5000

# 5. Serve the resulting global model + both web interfaces
uvicorn medsync.api.main:app --reload --app-dir src
#    http://localhost:8000/            -- diagnostic dashboard (upload an X-ray)
#    http://localhost:8000/monitoring  -- federation monitor (training history, node status)
```

## Docker

```powershell
docker compose up --build
```

Brings up an MLflow tracking server and one hospital-node diagnostic API
(`POST /predict` with a chest X-ray image, `GET /health`).

## Known limitations / next steps

- **Dataset size**: prototyped on the 5,606-image official Kaggle sample
  (363-2305/node after non-IID partitioning), not the full 112,120-image
  NIH set. The 0.558 macro AUC from experiment #3 is real but well below
  the project's 0.85 target — do not quote it as the final headline metric.
- **Single machine**: all 5 "hospital nodes" are simulated in one process
  on one GPU. Real deployment needs one node per physical/cloud location.
- **NVFlare migration**: see "Alternatives considered" above.
- **Privacy accounting**: per-round epsilon is tracked per client via
  Opacus's accountant; a cross-round composition analysis (total epsilon
  spent over all rounds, not just per-round) is not yet implemented.

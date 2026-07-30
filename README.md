# MedSync — Federated AI Diagnostic Network

A federated learning platform that lets hospitals collaboratively train a
chest X-ray diagnostic model without any raw patient data leaving their
premises. Each simulated hospital node trains locally with differentially
private SGD (Opacus); only noised model weights are sent to the server,
which aggregates them with FedAvg.

## Status

This is a working prototype, not the final target system, but it now
trains end-to-end on the **full 112,120-image NIH ChestX-ray14 dataset**
(not a subset) with real, monotonically-improving results — see
experiment #4 below. Still well short of the proposal's 0.85 AUC target;
"Known limitations" covers what's between here and there.

Two web interfaces are live: a clinician-facing **diagnostic dashboard**
(`/`, upload an X-ray, get findings) and an ops-facing **federation
monitor** (`/monitoring`, per-round AUC/loss/privacy-budget charts and
per-node status), both served by the same FastAPI app, sharing one design
system (`static/style.css`, light/dark mode). Both are behind login now --
see "Platform features" below.

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

**4. Full 112,120-image NIH dataset, non-IID across 5 nodes** (node sizes
11,140-31,939 after partitioning; 8 rounds x 1 local epoch, batch size
32, `scripts/convert_full_nih_dataset.py` + `scripts/run_federation.py`):

```
round  global_macro_auc  max_client_epsilon
1      0.5241            0.403
2      0.5371            0.403
3      0.5507            0.403
4      0.5548            0.403
5      0.5599            0.403
6      0.5649            0.403
7      0.5688            0.403
8      0.5702            0.403
```

Clean, monotonic improvement every single round (no oscillation, unlike
the smaller-dataset runs) -- **macro AUC reached 0.5702**, with the privacy
budget holding at ε≤0.403 throughout (tighter than the ε≤1.0 target, since
the much larger per-node datasets need less noise to hit the same target
epsilon). Read honestly: the climb hasn't plateaued -- only 8 total passes
over each node's data happened here (federated rounds x local epochs), far
fewer than the 20-50+ epochs published CheXNet-style results typically use
to reach their higher AUC numbers, and DP-SGD noise slows convergence
further as a deliberate privacy/utility tradeoff. More training rounds
would very likely keep improving this number; this run was stopped at 8
rounds as the reported result rather than chasing the 0.85 target further
in this session -- see "Known limitations".

## Stack

- **Model**: MONAI's `DenseNet121` (medical-imaging-oriented model library),
  CheXNet-style multi-label head, ImageNet-pretrained
- **Federation**: two orchestrators, both real, both wired to the same
  DP-SGD client logic (`src/medsync/federation/client.py`):
  - a custom FedAvg simulator (`scripts/run_federation.py`) -- simple,
    fully GPU-verified, the default/recommended path
  - an actual NVIDIA FLARE job (`scripts/run_nvflare_job.py` +
    `src/medsync/federation/nvflare_controller.py`) -- see "NVFlare
    integration" below for what that took and its current constraints
- **Privacy**: Opacus DP-SGD per client, per-round epsilon tracked and logged
- **Tracking**: MLflow
- **Serving**: FastAPI, two interfaces -- diagnostic dashboard (`/`) and
  federation monitor (`/monitoring`)
- **Platform**: JWT auth (login-gated `/predict`) + SQLite (`medsync.db`,
  via SQLAlchemy) for user accounts and a prediction log -- separate from
  MLflow, which stays the source of truth for training runs
- **Deployment**: Docker Compose (hospital node + MLflow server); a
  separate Dockerfile (`docker/Dockerfile.nvflare`) for the NVFlare path

## Platform features

Beyond the ML pipeline itself, there's a real (if intentionally minimal)
application layer:

- **Authentication**: JWT-based login (`/auth/login`), required for
  `/predict`. No public self-registration -- accounts are created by an
  admin via `scripts/create_user.py`, matching how real clinical software
  is provisioned (a hospital's IT admin creates staff accounts; there's no
  "sign up" button on a system that queries patient diagnostics).
- **Database**: SQLite (`medsync.db`) via SQLAlchemy, holding `users` and
  a `prediction_log` (who queried the model, what the top finding was,
  how long inference took). This is deliberately separate from MLflow --
  MLflow tracks *training* runs, this tracks *serving-time* usage.
- **Design system**: one shared stylesheet (`static/style.css`) across all
  three pages (diagnostic dashboard, federation monitor, login), with
  CSS-variable-based light/dark mode following the OS preference, rather
  than three pages independently reinventing colors and spacing.

```powershell
# Create an account (prompts for password if omitted -- don't put real
# passwords on the command line, they end up in shell history)
python scripts/create_user.py --username dr.smith --role clinician
```

The JWT signing secret defaults to an insecure placeholder for local dev
(`medsync/auth.py`); set a real one via the `MEDSYNC_JWT_SECRET`
environment variable for anything beyond your own machine -- Docker
Compose below requires it.

## NVFlare integration

NVIDIA FLARE is genuinely wired in now, not just installed unused: the
model, the DP-SGD client training, and the FedAvg aggregation are all
identical to the custom simulator (same `LocalClient`, same
`build_chexnet`/`CheXNet`), just orchestrated by NVFlare's Job API
(`MedSyncFedJob`/`MedSyncFedAvg` in `nvflare_controller.py`) and its local
simulator instead of a hand-rolled Python loop. Getting there surfaced
eight real, separate bugs, each fixed rather than worked around
superficially:

1. NVFlare's model validator requires the training module to already be in
   `.train()` mode before wrapping -- irrelevant to NVFlare specifically,
   this one also affected the custom simulator (fixed in `client.py`).
2. MONAI's `DenseNet121` needs three required constructor args
   (`spatial_dims`, `in_channels`, `out_channels`), but NVFlare's Job API
   reconstructs registered model components by calling `ClassName()` with
   no arguments (it serializes a JSON config, it doesn't pickle the live
   object) -- fixed by making `CheXNet` a real subclass with everything
   defaulted, and folding the GroupNorm/Opacus fix into `__init__` itself
   so a bare `CheXNet()` reconstruction is still fully consistent with
   every other copy in the system.
3. Passing an already-built `Dataset` into a controller's constructor
   fails the same way ("Object of type Subset is not JSON serializable")
   -- fixed by passing plain strings/ints/floats and building the eval
   set lazily inside `run()`.
4. NVFlare deploys and runs each client script from a per-site job
   workspace directory, not the repo root -- relative paths like
   `data/partitions` silently resolve against the wrong cwd there. Fixed
   by resolving to absolute paths once, in the top-level job script, and
   passing them explicitly via `script_args`.
5. **Windows-only**: `nvflare/fuel/f3/cellnet/net_agent.py` does an
   unconditional `import resource` (POSIX-only, used by an admin
   diagnostics command this project never calls) -- native Windows Python
   can't even import `nvflare` without a shim (`medsync/_win_compat.py`).
6. **Windows-only, unfixable by a shim**: NVFlare's simulator launches
   subprocesses via `subprocess.Popen(..., preexec_fn=os.setsid, ...)` --
   `os.setsid` doesn't exist on Windows, and Python's `subprocess.Popen`
   doesn't support `preexec_fn` on Windows at all. This is a hard platform
   wall, not something a compatibility stub can paper over.
7. NVFlare deploys and runs the *controller* (not just client scripts)
   from its own per-job workspace directory too -- MLflow's default
   `./mlruns` silently resolved there instead of this project's real
   `mlruns/`, so training metrics were being logged somewhere invisible to
   `/monitoring`. Fixed with an explicit absolute `mlflow.set_tracking_uri()`
   passed in from the top-level job script, same fix pattern as #4.
8. NVFlare's built-in weighted-average aggregator hands back **numpy
   arrays**, not torch tensors, regardless of the `PYTORCH` exchange
   format each client sends (`load_state_dict` then fails outright:
   "expected torch.Tensor ... but received numpy.ndarray"). This one only
   shows up on round 2+ if unfixed -- round 1 has nothing to aggregate
   from a prior round, so it's easy to miss testing with `--rounds 1`.
   Fixed by converting back to torch right after aggregation, in place,
   since the same object is what gets broadcast to clients next round.

Because of #6, **the NVFlare path only runs on Linux/Mac, or Docker/WSL2
on Windows** -- it cannot run via native Windows Python, unlike everything
else in this repo. `docker/Dockerfile.nvflare` + the commands below are
the verified way to run it from Windows:

```powershell
docker build -f docker/Dockerfile.nvflare -t medsync-nvflare .
docker run --rm `
  -v ${PWD}/data/partitions:/app/data/partitions:ro `
  -v ${PWD}/mlruns:/app/mlruns `
  medsync-nvflare --rounds 5 --device cpu --experiment medsync-nvflare
```

(`--device cpu`: this Docker setup has no GPU passthrough configured, so
NVFlare-orchestrated runs are CPU-only. This turned out to matter a lot:
a run against the full 5,606-image partitioned dataset took upwards of
20+ hours without completing a single round -- Opacus's per-sample-gradient
overhead on a 121-layer model, uninaccelerated, at that data volume, in a
resource-constrained container, is genuinely impractical, not hung. **The
integration itself is verified correct** via a fast synthetic smoke test
instead: 5 nodes x 6 tiny images, 2 full rounds, completed in under 5
minutes with correct MLflow logging and correct weight aggregation across
both rounds (`FINISHED` status, visible in `/monitoring`). Getting
practical speed out of the NVFlare path needs the NVIDIA Container Toolkit
configured on top of Docker Desktop's WSL2 backend for real GPU
passthrough -- not attempted in this session.)

Given the custom simulator is simpler, already GPU-verified at full data
scale, and produces identical training semantics, it remains the
recommended path for actually generating results (see "Experiment
history" -- all real numbers there come from the custom simulator on
GPU); the NVFlare path exists to demonstrate real integration with the
framework named in the original proposal, and as the natural next step if
this ever needs true multi-machine, cross-institution deployment (secure
provisioning, TLS, admin console) rather than one-process simulation.

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Requires an NVIDIA GPU (tested on RTX 3060, 6GB VRAM) with CUDA 12.1 drivers.

## Running the pipeline

```powershell
# 1. Get data -- three options:

#    (a) Full 112,120-image NIH dataset via Kaggle (recommended, and what
#        experiment #4's real results use). ~45GB download, ~90GB peak
#        disk during extraction on a SEPARATE drive if your project drive
#        is tight -- delete the raw zip/PNGs afterward, the converted
#        output is only ~2-3GB. Needs a Kaggle API token at
#        C:\Users\<you>\.kaggle\access_token (kaggle.com -> Settings ->
#        API -> Create New Token).
kaggle datasets download -d nih-chest-xrays/data -p <some-dir>
# unzip it (12 images_NNN/ subfolders + Data_Entry_2017.csv), then:
python scripts/convert_full_nih_dataset.py --kaggle-dir <extracted-dir>

#    (a2) Or just the 5,606-image sample for a much faster sanity check
#         (~2 minutes total, used for experiments #1-3):
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

# 3. Run federated training (FedAvg + DP-SGD), logs to MLflow.
#    Experiment #4's real numbers used these settings on the full dataset;
#    more rounds/local-epochs would likely keep improving AUC (see README
#    "Known limitations") at the cost of longer training time.
python scripts/run_federation.py --rounds 8 --local-epochs 1 --batch-size 32

# 4. Inspect metrics
mlflow ui   # http://localhost:5000

# 5. Create a login (prompts for password) -- required once, accounts persist in medsync.db
python scripts/create_user.py --username dr.smith --role clinician

# 6. Serve the resulting global model + both web interfaces
uvicorn medsync.api.main:app --reload --app-dir src
#    http://localhost:8000/login       -- sign in first
#    http://localhost:8000/            -- diagnostic dashboard (upload an X-ray)
#    http://localhost:8000/monitoring  -- federation monitor (training history, node status)
```

## Docker

```powershell
# SQLite bind-mounts need the file to already exist on the host, or Docker
# creates a directory at that path instead (breaking it) -- one-time setup:
New-Item -ItemType File -Path medsync.db -Force

$env:MEDSYNC_JWT_SECRET = "some-random-string-not-the-dev-default"
docker compose up --build
```

Brings up an MLflow tracking server and one hospital-node diagnostic API
(`POST /predict` with a chest X-ray image, `GET /health`).

## Known limitations / next steps

- **Training duration**: now trains on the full 112,120-image NIH set
  (experiment #4), so dataset size is no longer the bottleneck it was in
  experiments #1-3. The 0.5702 macro AUC is real, current, and climbing
  cleanly every round -- but only 8 rounds x 1 local epoch have run so
  far. The clear next lever is more rounds/epochs (30-50+, matching what
  published non-federated, non-private CheXNet results use), not more
  data. Do not quote 0.5702 as a ceiling -- it's a checkpoint mid-climb,
  not a plateau.
- **Single machine**: all 5 "hospital nodes" are simulated in one process
  on one GPU. Real deployment needs one node per physical/cloud location.
- **NVFlare on Windows**: the NVFlare-orchestrated path requires Linux/Mac
  or Docker/WSL2 -- see "NVFlare integration" above for exactly why.
- **Privacy accounting**: per-round epsilon is tracked per client via
  Opacus's accountant; a cross-round composition analysis (total epsilon
  spent over all rounds, not just per-round) is not yet implemented.

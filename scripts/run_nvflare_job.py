"""
Runs the federated training as an actual NVIDIA FLARE job (local simulator
mode -- all 5 simulated hospital "sites" run in one process/machine, same
as scripts/run_federation.py's custom simulator, but orchestrated by
NVFlare's Job API + FedAvg controller instead of a hand-rolled loop).

Usage:
    python scripts/run_nvflare_job.py --rounds 5
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import medsync._win_compat  # noqa: E402,F401
from medsync.federation.nvflare_controller import MedSyncFedJob  # noqa: E402
from medsync.models.chexnet import build_chexnet  # noqa: E402

from nvflare.job_config.script_runner import ScriptRunner  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions-dir", default="data/partitions")
    parser.add_argument("--num-nodes", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-epsilon", type=float, default=1.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.2)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workspace", default="nvflare_workspace")
    parser.add_argument("--experiment", default="medsync-nvflare")
    args = parser.parse_args()

    # Resolve to an absolute path once, here, while cwd is still the repo
    # root: both the controller (runs server-side) and each client script
    # (deployed into a per-site job workspace dir) need this, and a bare
    # relative path silently resolves against the wrong cwd in either place.
    partitions_dir_abs = str(Path(args.partitions_dir).resolve())
    checkpoint_path_abs = str((Path.cwd() / "global_model_final.pth").resolve())
    mlflow_tracking_uri = Path("mlruns").resolve().as_uri()

    initial_model = build_chexnet(pretrained=True)

    job = MedSyncFedJob(
        initial_model=initial_model,
        n_clients=args.num_nodes,
        num_rounds=args.rounds,
        partitions_dir=partitions_dir_abs,
        val_split=args.val_split,
        device=args.device,
        name=args.experiment,
        run_params=vars(args),
        checkpoint_path=checkpoint_path_abs,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    client_args = (
        f"--partitions-dir {partitions_dir_abs} --val-split {args.val_split} "
        f"--local-epochs {args.local_epochs} --batch-size {args.batch_size} --lr {args.lr} "
        f"--target-epsilon {args.target_epsilon} --target-delta {args.target_delta} "
        f"--max-grad-norm {args.max_grad_norm}"
    )
    for i in range(args.num_nodes):
        executor = ScriptRunner(script="scripts/nvflare_client_train.py", script_args=client_args)
        job.to(executor, f"site-{i + 1}")

    # threads=1: keep clients sequential rather than concurrent on one GPU --
    # matches the custom simulator's behavior and avoids untested concurrent
    # CUDA access from multiple Opacus-wrapped models at once.
    job.simulator_run(args.workspace, threads=1)


if __name__ == "__main__":
    main()

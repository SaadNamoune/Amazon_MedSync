"""
NVFlare "site" script: runs inside NVFlare's client executor (one instance
per simulated hospital node). This is the actual federation participant --
it never sees another node's data, only receives global model weights and
sends back its own locally-trained (DP-noised) weights.

Reuses the same DP-SGD LocalClient from src/medsync/federation/client.py
that the custom simulator (scripts/run_federation.py) uses, so the local
training + privacy-calibration logic is identical either way; only the
orchestration (who talks to whom, how rounds are driven) changes.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import medsync._win_compat  # noqa: E402,F401
import nvflare.client as flare  # noqa: E402

from medsync.data.dataset import split_train_val  # noqa: E402
from medsync.federation.client import LocalClient  # noqa: E402
from medsync.federation.evaluate import evaluate_model  # noqa: E402
from medsync.models.chexnet import build_chexnet  # noqa: E402


def main():
    # NVFlare deploys and runs this script from a per-site job workspace
    # directory (e.g. nvflare_workspace/site-1/simulate_job/app_site-1/custom/),
    # not the repo root -- a relative "data/partitions" path silently resolves
    # to the wrong place there. The caller (run_nvflare_job.py) passes an
    # absolute path explicitly instead of relying on any particular cwd.
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions-dir", required=True)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-epsilon", type=float, default=1.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.2)
    args = parser.parse_args()

    flare.init()
    site_name = flare.get_site_name()  # e.g. "site-1"
    node_idx = int(site_name.split("-")[-1]) - 1
    node_dir = Path(args.partitions_dir) / f"node_{node_idx}"

    train_ds, val_ds = split_train_val(node_dir, val_split=args.val_split)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = LocalClient(
        node_id=node_idx, dataset=train_ds, batch_size=args.batch_size, lr=args.lr,
        target_epsilon=args.target_epsilon, target_delta=args.target_delta,
        max_grad_norm=args.max_grad_norm, device=device,
    )
    print(f"[{site_name} -> node_{node_idx}] {len(train_ds)} train / {len(val_ds)} held-out images")

    while flare.is_running():
        input_model = flare.receive()

        model = build_chexnet(pretrained=False)
        model.load_state_dict(input_model.params)

        state_dict, n_samples, epsilon, avg_loss = client.train_round(model, local_epochs=args.local_epochs)

        model.load_state_dict(state_dict)
        local_auc, _ = evaluate_model(model, val_ds, device=device)
        print(f"[{site_name}] round {input_model.current_round}: "
              f"loss={avg_loss:.4f} epsilon={epsilon:.3f} local_auc={local_auc:.4f}")

        output_model = flare.FLModel(
            params=state_dict,
            metrics={"loss": avg_loss, "epsilon": epsilon, "local_auc": local_auc},
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()

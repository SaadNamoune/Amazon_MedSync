"""
Runs the full federated training simulation: 5 hospital nodes train locally
with DP-SGD, a server FedAvg-aggregates their weights each round, and the
global model is evaluated on a held-out set after every round.

Usage:
    python scripts/run_federation.py --rounds 5 --local-epochs 1
"""
import argparse
import copy
import sys
from pathlib import Path

import mlflow
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medsync.data.dataset import ChestXrayDataset  # noqa: E402
from medsync.models.chexnet import build_chexnet  # noqa: E402
from medsync.federation.client import LocalClient  # noqa: E402
from medsync.federation.fedavg import federated_average  # noqa: E402
from medsync.federation.evaluate import evaluate_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions-dir", default="data/partitions")
    parser.add_argument("--num-nodes", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.2)
    parser.add_argument("--val-split", type=float, default=0.15,
                         help="Fraction of each node's data held out for the global eval set")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--experiment", default="medsync-federated")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)

    partitions_dir = Path(args.partitions_dir)
    clients, held_out_datasets = [], []
    for node_id in range(args.num_nodes):
        node_dir = partitions_dir / f"node_{node_id}"
        full_ds = ChestXrayDataset(node_dir, train=True)
        n_val = max(1, int(len(full_ds) * args.val_split))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        held_out_datasets.append(val_ds)
        clients.append(LocalClient(
            node_id=node_id, dataset=train_ds, batch_size=args.batch_size,
            lr=args.lr, noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm, device=args.device,
        ))
        print(f"node_{node_id}: {n_train} train / {n_val} held-out eval images")

    global_eval_ds = torch.utils.data.ConcatDataset(held_out_datasets)

    global_model = build_chexnet(pretrained=True)

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        for round_idx in range(1, args.rounds + 1):
            print(f"\n=== Round {round_idx}/{args.rounds} ===")
            state_dicts, weights, epsilons, losses = [], [], [], []

            for client in clients:
                local_model = copy.deepcopy(global_model)
                local_model.load_state_dict(global_model.state_dict())
                state_dict, n_samples, epsilon, avg_loss = client.train_round(
                    local_model, local_epochs=args.local_epochs
                )
                state_dicts.append(state_dict)
                weights.append(n_samples)
                epsilons.append(epsilon)
                losses.append(avg_loss)
                print(f"  node_{client.node_id}: loss={avg_loss:.4f} "
                      f"epsilon={epsilon:.3f} n={n_samples}")

            new_state = federated_average(state_dicts, weights)
            global_model.load_state_dict(new_state)

            macro_auc, per_label_auc = evaluate_model(
                global_model, global_eval_ds, device=args.device
            )
            max_epsilon = max(epsilons)
            avg_client_loss = sum(losses) / len(losses)

            print(f"  -> global macro AUC={macro_auc:.4f} "
                  f"max_client_epsilon={max_epsilon:.3f}")

            mlflow.log_metrics({
                "macro_auc": macro_auc,
                "avg_client_loss": avg_client_loss,
                "max_client_epsilon": max_epsilon,
            }, step=round_idx)

        ckpt_path = Path("global_model_final.pth")
        torch.save(global_model.state_dict(), ckpt_path)
        mlflow.log_artifact(str(ckpt_path))
        print(f"\nSaved final global model -> {ckpt_path}")


if __name__ == "__main__":
    main()

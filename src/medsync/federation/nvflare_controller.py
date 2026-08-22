"""
Custom NVFlare FedAvg controller: adds the one thing NVFlare's built-in
FedAvg doesn't do out of the box -- evaluating the newly-aggregated global
model against a held-out set pooled across all hospital nodes, and logging
round-by-round metrics to MLflow so the existing /monitoring dashboard
(which reads from MLflow) works unchanged regardless of which orchestrator
produced the run.
"""
from collections import OrderedDict
from typing import List, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn

import medsync._win_compat  # noqa: F401
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver

from medsync.data.dataset import load_pooled_eval_dataset
from medsync.federation.evaluate import evaluate_model
from medsync.models.chexnet import build_chexnet


def _to_torch_state_dict(params: dict) -> OrderedDict:
    """NVFlare's built-in weighted-average aggregator (WeightedAggregationHelper)
    operates on and returns numpy arrays regardless of the PYTORCH exchange
    format each client sends -- load_state_dict() rejects those outright
    ("expected torch.Tensor ... but received numpy.ndarray"), so the
    aggregated params need converting back before they're usable again."""
    return OrderedDict(
        (k, torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
        for k, v in params.items()
    )


class MedSyncFedAvg(FedAvg):
    """FedAvg controller that also evaluates the aggregated global model
    each round (on a held-out set pooled across all hospital nodes) and
    logs to MLflow -- NVFlare's built-in FedAvg does neither, it only
    averages client weights. The mlflow run is opened/closed inside run()
    itself (not by the caller) because NVFlare's simulator runs the
    controller and each client's ScriptRunner in separate threads, and
    MLflow's active-run tracking doesn't reliably cross threads.

    Takes plain strings/ints/floats (partitions_dir, num_nodes, val_split)
    rather than a constructed Dataset, and builds the eval set lazily
    inside run() -- NVFlare serializes every component's constructor args
    to a JSON job config, so a live Dataset object here would fail with
    "Object of type Subset is not JSON serializable" the moment the job
    is built, not even when it's run.
    """

    def __init__(self, *args, partitions_dir: str = "data/partitions",
                 num_nodes: int = 5, val_split: float = 0.15, device: str = "cuda",
                 experiment_name: str = "medsync-nvflare", run_params: Optional[dict] = None,
                 checkpoint_path: str = "global_model_final.pth",
                 mlflow_tracking_uri: Optional[str] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.partitions_dir = partitions_dir
        self.num_nodes = num_nodes
        self.val_split = val_split
        self.device = device
        self.experiment_name = experiment_name
        self.run_params = run_params or {}
        self.checkpoint_path = checkpoint_path
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def run(self) -> None:
        self.info("Start MedSync FedAvg (NVFlare-orchestrated).")

        global_eval_dataset = load_pooled_eval_dataset(
            self.partitions_dir, self.num_nodes, self.val_split
        )

        # NVFlare runs the controller from its own per-job workspace
        # directory too (same issue as the client scripts) -- mlflow's
        # default "./mlruns" is relative to whatever cwd that happens to
        # be, silently writing tracking data somewhere inside the NVFlare
        # workspace instead of this project's real mlruns/, invisible to
        # the /monitoring dashboard. An explicit absolute URI avoids that.
        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            mlflow.log_params(self.run_params)

            model = self.load_model()
            model.start_round = self.start_round
            model.total_rounds = self.num_rounds

            for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
                self.info(f"Round {self.current_round} started.")
                model.current_round = self.current_round

                clients = self.sample_clients(self.num_clients)
                results = self.send_model_and_wait(targets=clients, data=model)

                aggregate_results = self.aggregate(results, aggregate_fn=self.aggregate_fn)
                model = self.update_model(model, aggregate_results)
                # Convert back to torch here, once, in place: NVFlare's built-in
                # weighted-average aggregator hands back numpy arrays regardless
                # of the PYTORCH exchange format clients send, and this same
                # `model` object is what gets broadcast to clients next round --
                # leaving it as numpy would make round 2 fail on the client side
                # the same way it just failed here.
                model.params = _to_torch_state_dict(model.params)
                self.save_model(model)

                eval_model = build_chexnet(pretrained=False)
                eval_model.load_state_dict(model.params)
                macro_auc, _ = evaluate_model(eval_model, global_eval_dataset, device=self.device)

                losses = [r.metrics.get("loss", 0.0) for r in results if r.metrics]
                epsilons = [r.metrics.get("epsilon", 0.0) for r in results if r.metrics]
                cumulative_epsilons = [r.metrics.get("cumulative_epsilon", 0.0) for r in results if r.metrics]
                avg_client_loss = sum(losses) / len(losses) if losses else float("nan")
                max_client_epsilon = max(epsilons) if epsilons else float("nan")
                cumulative_max_epsilon = max(cumulative_epsilons) if cumulative_epsilons else float("nan")

                self.info(f"Round {self.current_round}: global_macro_auc={macro_auc:.4f} "
                          f"avg_client_loss={avg_client_loss:.4f} max_client_epsilon={max_client_epsilon:.3f} "
                          f"cumulative_max_epsilon={cumulative_max_epsilon:.3f}")

                mlflow.log_metrics({
                    "macro_auc": macro_auc,
                    "avg_client_loss": avg_client_loss,
                    "max_client_epsilon": max_client_epsilon,
                    "cumulative_max_epsilon": cumulative_max_epsilon,
                }, step=self.current_round)

            torch.save(model.params, self.checkpoint_path)
            mlflow.log_artifact(self.checkpoint_path)

        self.info("Finished MedSync FedAvg.")


class MedSyncFedJob(BaseFedJob):
    """Job API wiring for MedSyncFedAvg -- mirrors nvflare's own FedAvgJob
    (app_opt/pt/job_config/fed_avg.py) but plugs in our controller instead
    of the stock one, since FedAvgJob hardcodes the stock FedAvg."""

    def __init__(
        self,
        initial_model: nn.Module,
        n_clients: int,
        num_rounds: int,
        partitions_dir: str = "data/partitions",
        val_split: float = 0.15,
        device: str = "cuda",
        name: str = "medsync-nvflare",
        key_metric: str = "local_auc",
        run_params: Optional[dict] = None,
        checkpoint_path: str = "global_model_final.pth",
        mlflow_tracking_uri: Optional[str] = None,
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
    ):
        # Use NVFlare's MLflow analytics receiver instead of the default
        # TensorBoard one: this project already standardizes on MLflow for
        # tracking (see /api/training-history), and tensorboard's protobuf
        # requirement (>=6.31) hard-conflicts with mlflow-skinny's (<6) in
        # this environment -- no version of protobuf satisfies both.
        super().__init__(
            initial_model, name, min_clients, mandatory_clients, key_metric,
            analytics_receiver=MLflowReceiver(kw_args={"experiment_name": name}),
        )

        controller = MedSyncFedAvg(
            num_clients=n_clients,
            num_rounds=num_rounds,
            persistor_id=self.comp_ids["persistor_id"],
            partitions_dir=partitions_dir,
            num_nodes=n_clients,
            val_split=val_split,
            device=device,
            experiment_name=name,
            run_params=run_params,
            checkpoint_path=checkpoint_path,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        self.to_server(controller)

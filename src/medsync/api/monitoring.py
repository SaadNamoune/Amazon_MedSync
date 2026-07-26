"""
Federation monitoring endpoints: training history (from MLflow) and
hospital node status (from the partitioned data on disk). This is the
"who's participating and how well is it training" view, as opposed to
main.py's clinician-facing single-image prediction endpoint.
"""
import csv
import math
from pathlib import Path

from typing import Optional

from fastapi import APIRouter, HTTPException
from mlflow.tracking import MlflowClient

router = APIRouter(prefix="/api")

PARTITIONS_DIR = Path("data/partitions")
METRIC_KEYS = ["macro_auc", "avg_client_loss", "max_client_epsilon"]


def _json_safe(value):
    """Starlette's JSONResponse renders with allow_nan=False (correctly,
    NaN/Infinity aren't valid JSON), so a NaN metric -- which happens for
    real on degenerate data, e.g. an eval set too small for any label to
    have both classes present -- crashes the whole response with a 500
    instead of just... being null, which is what it actually means here."""
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@router.get("/experiments")
def list_experiments():
    client = MlflowClient()
    experiments = []
    for exp in client.search_experiments(order_by=["creation_time DESC"]):
        if exp.name == "Default":
            continue
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1,
                                    order_by=["start_time DESC"])
        experiments.append({
            "name": exp.name,
            "experiment_id": exp.experiment_id,
            "latest_run_id": runs[0].info.run_id if runs else None,
        })
    return experiments


@router.get("/training-history")
def training_history(experiment: Optional[str] = None):
    client = MlflowClient()

    if experiment is not None:
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            raise HTTPException(404, f"No such experiment: {experiment}")
        exp_ids = [exp.experiment_id]
    else:
        # No experiment given: show the most recent run across all of them,
        # so the dashboard surfaces the latest result without the caller
        # needing to know experiment names in advance.
        exp_ids = [e.experiment_id for e in client.search_experiments() if e.name != "Default"]
        if not exp_ids:
            raise HTTPException(404, "No experiments logged yet")

    runs = client.search_runs(experiment_ids=exp_ids,
                               order_by=["start_time DESC"], max_results=1)
    if not runs:
        raise HTTPException(404, f"No runs logged yet for experiment: {experiment}")
    run = runs[0]

    histories = {key: client.get_metric_history(run.info.run_id, key) for key in METRIC_KEYS}
    steps = sorted({m.step for m in histories["macro_auc"]})
    by_step = {key: {m.step: m.value for m in hist} for key, hist in histories.items()}

    rounds = [
        {"round": step, **{key: _json_safe(by_step[key].get(step)) for key in METRIC_KEYS}}
        for step in steps
    ]

    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "params": run.data.params,
        "rounds": rounds,
    }


@router.get("/nodes")
def node_status():
    if not PARTITIONS_DIR.exists():
        return []

    nodes = []
    for node_dir in sorted(PARTITIONS_DIR.glob("node_*")):
        labels_csv = node_dir / "labels.csv"
        if not labels_csv.exists():
            continue

        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        label_counts = {}
        for row in rows:
            for label, value in row.items():
                if label == "filename" or value != "1":
                    continue
                label_counts[label] = label_counts.get(label, 0) + 1
        top_labels = sorted(label_counts.items(), key=lambda kv: -kv[1])[:3]

        nodes.append({
            "node_id": node_dir.name,
            "num_images": len(rows),
            "top_labels": [{"label": label, "count": count} for label, count in top_labels],
        })
    return nodes

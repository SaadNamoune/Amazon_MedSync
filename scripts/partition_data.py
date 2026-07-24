"""
Splits data/raw into N simulated hospital nodes with a non-IID label
distribution (Dirichlet allocation), so nodes look like real hospitals with
different case mixes rather than identical random samples -- this is what
makes the federated-averaging result meaningful to report.
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No_Finding",
]


def dominant_label(row):
    """Pick one label per image (its first positive finding, else No_Finding)
    to drive the Dirichlet split -- images stay multi-label in the output CSV."""
    for i, name in enumerate(LABEL_NAMES):
        if row[name] == "1":
            return name
    return "No_Finding"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/partitions")
    parser.add_argument("--num-nodes", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5,
                         help="Dirichlet concentration; lower = more skewed/non-IID")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    with open(src_dir / "labels.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_label = defaultdict(list)
    for row in rows:
        by_label[dominant_label(row)].append(row)

    node_rows = [[] for _ in range(args.num_nodes)]
    for label, label_rows in by_label.items():
        rng.shuffle(label_rows)
        proportions = rng.dirichlet(alpha=[args.alpha] * args.num_nodes)
        counts = (proportions * len(label_rows)).astype(int)
        counts[-1] = len(label_rows) - counts[:-1].sum()  # fix rounding drift

        idx = 0
        for node_id, count in enumerate(counts):
            node_rows[node_id].extend(label_rows[idx: idx + count])
            idx += count

    for node_id in range(args.num_nodes):
        node_dir = out_dir / f"node_{node_id}"
        node_img_dir = node_dir / "images"
        node_img_dir.mkdir(parents=True, exist_ok=True)

        rows_for_node = node_rows[node_id]
        with open(node_dir / "labels.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename"] + LABEL_NAMES)
            for row in rows_for_node:
                shutil.copy2(src_dir / "images" / row["filename"],
                             node_img_dir / row["filename"])
                writer.writerow([row["filename"]] +
                                 [row[name] for name in LABEL_NAMES])

        label_counts = defaultdict(int)
        for row in rows_for_node:
            label_counts[dominant_label(row)] += 1
        print(f"node_{node_id}: {len(rows_for_node)} images | "
              f"top labels: {sorted(label_counts.items(), key=lambda x: -x[1])[:3]}")

    print(f"\nPartitioned {len(rows)} images across {args.num_nodes} nodes -> {out_dir}")


if __name__ == "__main__":
    main()

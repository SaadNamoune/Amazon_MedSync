"""
Streams a bounded subset of the NIH ChestX-ray14 dataset from the Hugging Face
Hub (alkzar90/NIH-Chest-X-ray-dataset) and materializes it locally.

We stream instead of calling load_dataset() plainly because the full dataset
is 45GB; for a federated-learning prototype across 5 simulated hospital nodes
we only need a few thousand images to get a real, measurable signal.
"""
import argparse
import csv
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image

LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No_Finding",
]

IMAGE_SIZE = 224


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--num-images", type=int, default=3000)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming '{args.split}' split from alkzar90/NIH-Chest-X-ray-dataset ...")
    ds = load_dataset(
        "alkzar90/NIH-Chest-X-ray-dataset",
        "image-classification",
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )

    rows = []
    n_written = 0
    for i, example in enumerate(ds):
        if n_written >= args.num_images:
            break
        image: Image.Image = example["image"]
        labels = example["labels"]  # list of int label indices

        fname = f"img_{n_written:06d}.jpg"
        image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)).save(
            img_dir / fname, quality=90
        )

        multi_hot = [0] * len(LABEL_NAMES)
        for lbl in labels:
            if 0 <= lbl < len(LABEL_NAMES):
                multi_hot[lbl] = 1
        rows.append([fname] + multi_hot)
        n_written += 1

        if n_written % 250 == 0:
            print(f"  ... {n_written}/{args.num_images} images saved")

    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + LABEL_NAMES)
        writer.writerows(rows)

    print(f"Done. {n_written} images -> {img_dir}")
    print(f"Labels -> {csv_path}")


if __name__ == "__main__":
    main()

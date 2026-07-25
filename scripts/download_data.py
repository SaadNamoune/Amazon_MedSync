"""
Streams a bounded subset of the NIH ChestX-ray14 dataset from the Hugging Face
Hub (alkzar90/NIH-Chest-X-ray-dataset) and materializes it locally.

We stream instead of calling load_dataset() plainly because the full dataset
is 45GB; for a federated-learning prototype across 5 simulated hospital nodes
we only need a few thousand images to get a real, measurable signal.
"""
import argparse
import csv
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
    csv_path = out_dir / "labels.csv"

    # Resume support: the streaming split has a stable, deterministic iteration
    # order (no shuffling), so re-running just re-produces the same first N
    # examples. Skip past whatever we already saved instead of re-downloading
    # it -- that first batch is the expensive part (NIH's archive is slow).
    n_already = 0
    if csv_path.exists():
        with open(csv_path) as f:
            n_already = sum(1 for _ in f) - 1  # minus header
        n_already = max(n_already, 0)
        print(f"Resuming: {n_already} images already saved, skipping those in the stream")

    print(f"Streaming '{args.split}' split from alkzar90/NIH-Chest-X-ray-dataset ...")
    ds = load_dataset(
        "alkzar90/NIH-Chest-X-ray-dataset",
        "image-classification",
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )
    if n_already:
        ds = ds.skip(n_already)

    # Written incrementally (not buffered until the end) so an interrupted
    # streaming run -- these can take a long time over a slow remote archive
    # -- still leaves a usable, complete labels.csv for whatever finished.
    n_written = n_already
    mode = "a" if n_already else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if not n_already:
            writer.writerow(["filename"] + LABEL_NAMES)

        for example in ds:
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
            writer.writerow([fname] + multi_hot)
            f.flush()
            n_written += 1

            if n_written % 50 == 0:
                print(f"  ... {n_written}/{args.num_images} images saved", flush=True)

    print(f"Done. {n_written} images -> {img_dir}")
    print(f"Labels -> {csv_path}")


if __name__ == "__main__":
    main()

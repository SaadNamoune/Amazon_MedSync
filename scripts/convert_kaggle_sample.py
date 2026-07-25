"""
Converts the official NIH ChestX-ray14 Kaggle sample
(kaggle datasets download -d nih-chest-xrays/sample) into this project's
data/raw format (resized JPEGs + a labels.csv matching LABEL_NAMES).

This is an alternative to scripts/download_data.py's HF streaming path --
Kaggle serves the same 5,606-image sample as one bulk zip, which downloads
in ~2 minutes instead of the ~13+ hours HF's row-by-row remote-archive
streaming would take for the same image count.
"""
import argparse
import csv
from pathlib import Path

from PIL import Image

LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No_Finding",
]

IMAGE_SIZE = 224


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", required=True,
                         help="Path to extracted nih-chest-xrays/sample (contains sample_labels.csv and sample/images/)")
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args()

    kaggle_dir = Path(args.kaggle_dir)
    src_csv = kaggle_dir / "sample_labels.csv"
    src_img_dir = kaggle_dir / "sample" / "images"

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    with open(src_csv) as f:
        rows = list(csv.DictReader(f))

    csv_path = out_dir / "labels.csv"
    n_written = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + LABEL_NAMES)

        for row in rows:
            findings = row["Finding Labels"].split("|")
            multi_hot = [0] * len(LABEL_NAMES)
            for finding in findings:
                name = finding.strip().replace(" ", "_")
                if name in LABEL_NAMES:
                    multi_hot[LABEL_NAMES.index(name)] = 1

            src_path = src_img_dir / row["Image Index"]
            if not src_path.exists():
                continue

            fname = f"img_{n_written:06d}.jpg"
            Image.open(src_path).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ).save(img_dir / fname, quality=90)

            writer.writerow([fname] + multi_hot)
            n_written += 1

            if n_written % 500 == 0:
                print(f"  ... {n_written}/{len(rows)} images converted", flush=True)

    print(f"Done. {n_written} images -> {img_dir}")
    print(f"Labels -> {csv_path}")


if __name__ == "__main__":
    main()

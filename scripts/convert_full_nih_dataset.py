"""
Converts the full NIH ChestX-ray14 dataset (112,120 images, downloaded via
`kaggle datasets download -d nih-chest-xrays/data`) into this project's
data/raw format (resized JPEGs + a labels.csv matching LABEL_NAMES).

Unlike the 5,606-image sample (scripts/convert_kaggle_sample.py, one flat
images/ folder), the full dataset ships as 12 separate images_NNN/images/
folders -- this script builds a filename index across all of them first.
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


def build_filename_index(root_dir: Path) -> dict:
    index = {}
    for folder in sorted(root_dir.glob("images_*/images")):
        for img_path in folder.glob("*.png"):
            index[img_path.name] = img_path
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", required=True,
                         help="Path to extracted nih-chest-xrays/data (contains Data_Entry_2017.csv and images_NNN/ folders)")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of images (for testing)")
    args = parser.parse_args()

    kaggle_dir = Path(args.kaggle_dir)
    src_csv = kaggle_dir / "Data_Entry_2017.csv"

    print("Indexing image files across images_001..images_012 ...")
    filename_index = build_filename_index(kaggle_dir)
    print(f"  found {len(filename_index)} image files")

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    with open(src_csv) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    csv_path = out_dir / "labels.csv"
    n_written = 0
    n_missing = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + LABEL_NAMES)

        for row in rows:
            src_path = filename_index.get(row["Image Index"])
            if src_path is None:
                n_missing += 1
                continue

            findings = row["Finding Labels"].split("|")
            multi_hot = [0] * len(LABEL_NAMES)
            for finding in findings:
                name = finding.strip().replace(" ", "_")
                if name in LABEL_NAMES:
                    multi_hot[LABEL_NAMES.index(name)] = 1

            fname = f"img_{n_written:06d}.jpg"
            Image.open(src_path).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ).save(img_dir / fname, quality=90)

            writer.writerow([fname] + multi_hot)
            n_written += 1

            if n_written % 2000 == 0:
                print(f"  ... {n_written}/{len(rows)} images converted", flush=True)

    print(f"Done. {n_written} images -> {img_dir} ({n_missing} rows had no matching image file)")
    print(f"Labels -> {csv_path}")


if __name__ == "__main__":
    main()

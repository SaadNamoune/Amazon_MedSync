import csv
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medsync.data.dataset import ChestXrayDataset, LABEL_NAMES  # noqa: E402


def _make_node_shard(tmp_path, n_images=3):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    for i in range(n_images):
        fname = f"img_{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 10, 0, 0)).save(img_dir / fname)
        labels = [1 if j == i % len(LABEL_NAMES) else 0 for j in range(len(LABEL_NAMES))]
        rows.append([fname] + labels)

    with open(tmp_path / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + LABEL_NAMES)
        writer.writerows(rows)
    return tmp_path


def test_dataset_length_and_shapes(tmp_path):
    node_dir = _make_node_shard(tmp_path, n_images=4)
    ds = ChestXrayDataset(node_dir, train=True)
    assert len(ds) == 4

    image, labels = ds[0]
    assert image.shape == (3, 32, 32)
    assert labels.shape == (len(LABEL_NAMES),)
    assert labels.sum().item() == 1  # exactly one positive label per synthetic row

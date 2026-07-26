import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No_Finding",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ChestXrayDataset(Dataset):
    """Reads a (images/, labels.csv) node shard produced by scripts/partition_data.py."""

    def __init__(self, node_dir: str, train: bool = True):
        self.node_dir = Path(node_dir)
        self.transform = build_transform(train)

        with open(self.node_dir / "labels.csv") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(self.node_dir / "images" / row["filename"]).convert("RGB")
        image = self.transform(image)
        labels = torch.tensor([float(row[name]) for name in LABEL_NAMES])
        return image, labels


def split_train_val(node_dir: str, val_split: float = 0.15, seed: int = 42):
    """Splits one node's shard into train/held-out-val, deterministically
    (fixed seed) so the same held-out images are used whether the caller is
    the custom simulator or the NVFlare-orchestrated job -- their results
    are only comparable if they evaluate on the same data."""
    full_ds = ChestXrayDataset(node_dir, train=True)
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    return torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )


def load_pooled_eval_dataset(partitions_dir: str, num_nodes: int, val_split: float = 0.15):
    """Held-out validation images pooled across all hospital nodes -- this
    is what "global macro AUC" is measured against."""
    from pathlib import Path as _Path

    held_out = []
    for node_id in range(num_nodes):
        node_dir = _Path(partitions_dir) / f"node_{node_id}"
        _, val_ds = split_train_val(node_dir, val_split=val_split)
        held_out.append(val_ds)
    return torch.utils.data.ConcatDataset(held_out)

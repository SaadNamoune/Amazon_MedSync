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

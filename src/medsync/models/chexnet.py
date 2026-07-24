import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

NUM_CLASSES = 15  # 14 pathologies + No_Finding, see data/dataset.py LABEL_NAMES


def build_chexnet(pretrained: bool = True) -> nn.Module:
    """DenseNet121 with a multi-label sigmoid head, matching the CheXNet
    architecture (Rajpurkar et al., 2017) used across all hospital nodes."""
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)
    return model

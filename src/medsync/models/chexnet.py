import torch.nn as nn
from opacus.validators import ModuleValidator
from torchvision.models import densenet121, DenseNet121_Weights

NUM_CLASSES = 15  # 14 pathologies + No_Finding, see data/dataset.py LABEL_NAMES


def build_chexnet(pretrained: bool = True) -> nn.Module:
    """DenseNet121 with a multi-label sigmoid head, matching the CheXNet
    architecture (Rajpurkar et al., 2017) used across all hospital nodes.

    BatchNorm is replaced with GroupNorm (via Opacus's ModuleValidator) because
    DP-SGD requires per-sample gradients, and BatchNorm mixes information
    across the batch dimension, which breaks that guarantee. This must happen
    here -- at the single shared build site -- rather than per-client, since
    every node's state_dict has to share the same architecture for FedAvg to
    aggregate them.
    """
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)
    model = ModuleValidator.fix(model)
    return model

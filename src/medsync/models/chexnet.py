from monai.networks.nets import DenseNet121
from opacus.validators import ModuleValidator

NUM_CLASSES = 15  # 14 pathologies + No_Finding, see data/dataset.py LABEL_NAMES


class CheXNet(DenseNet121):
    """DenseNet121 with a multi-label sigmoid head, matching the CheXNet
    architecture (Rajpurkar et al., 2017) used across all hospital nodes.

    Uses MONAI's DenseNet121 (medical-imaging-oriented model library) rather
    than torchvision's generic one -- same underlying architecture and the
    same ImageNet-pretrained weights under the hood, but MONAI's is the
    version built for this domain (consistent spatial_dims handling for
    2D/3D scans, used elsewhere in the medical imaging pipeline).

    This is a real subclass (not just a factory function returning a plain
    DenseNet121) with every constructor arg defaulted, so `CheXNet()` works
    with zero arguments. That's a hard requirement, not a style choice:
    NVFlare's Job API reconstructs registered model components by calling
    `ClassName()` from a JSON config it generates -- it does not pickle the
    live object -- so a class requiring positional args (MONAI's raw
    DenseNet121 needs spatial_dims/in_channels/out_channels) fails there
    with "missing required positional arguments" even though it works fine
    everywhere we construct it directly ourselves.

    The GroupNorm fix (below) is applied inside __init__ itself, not left
    to the caller, for the same reason: if NVFlare ever reconstructs a bare
    `CheXNet()`, it must come out GroupNorm-fixed like every other copy in
    the system, or its state_dict keys/shapes won't match models trained
    elsewhere and loading a checkpoint into it will fail.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__(
            spatial_dims=2, in_channels=3, out_channels=NUM_CLASSES, pretrained=pretrained
        )
        # BatchNorm is replaced with GroupNorm (via Opacus's ModuleValidator)
        # because DP-SGD requires per-sample gradients, and BatchNorm mixes
        # information across the batch dimension, breaking that guarantee.
        # ModuleValidator.fix() clones the module and returns the fixed
        # clone rather than mutating in place, so we transplant its state
        # onto self instead of returning a different object from __init__.
        fixed = ModuleValidator.fix(self)
        self.__dict__.update(fixed.__dict__)


def build_chexnet(pretrained: bool = True) -> CheXNet:
    return CheXNet(pretrained=pretrained)

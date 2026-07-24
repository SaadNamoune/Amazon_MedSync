import sys
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medsync.federation.fedavg import federated_average  # noqa: E402


def test_federated_average_equal_weights():
    sd_a = OrderedDict(w=torch.tensor([1.0, 2.0]))
    sd_b = OrderedDict(w=torch.tensor([3.0, 4.0]))
    result = federated_average([sd_a, sd_b], weights=[1, 1])
    assert torch.allclose(result["w"], torch.tensor([2.0, 3.0]))


def test_federated_average_weighted():
    sd_a = OrderedDict(w=torch.tensor([0.0]))
    sd_b = OrderedDict(w=torch.tensor([10.0]))
    # node_b has 3x the samples of node_a -> pulled towards node_b
    result = federated_average([sd_a, sd_b], weights=[1, 3])
    assert torch.allclose(result["w"], torch.tensor([7.5]))

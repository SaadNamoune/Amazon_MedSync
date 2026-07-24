from collections import OrderedDict
from typing import List, Tuple

import torch


def federated_average(
    state_dicts: List[OrderedDict], weights: List[int]
) -> OrderedDict:
    """Weighted FedAvg: each client's state_dict is weighted by its local
    dataset size, matching McMahan et al. (2017)."""
    total = sum(weights)
    avg_state = OrderedDict()
    for key in state_dicts[0].keys():
        stacked = torch.stack(
            [sd[key].float() * (w / total) for sd, w in zip(state_dicts, weights)]
        )
        avg_state[key] = stacked.sum(dim=0).to(state_dicts[0][key].dtype)
    return avg_state

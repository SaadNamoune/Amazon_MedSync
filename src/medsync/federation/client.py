from collections import OrderedDict

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader


class LocalClient:
    """One simulated hospital node: trains the global model on its own
    private shard with DP-SGD (Opacus) so raw gradients never leave the
    node in a form that could be reverse-engineered to patient data.

    The noise multiplier is calibrated per round (via Opacus's privacy
    accountant) to hit `target_epsilon`, rather than fixed -- a fixed
    noise_multiplier gives wildly different actual epsilon depending on
    dataset size and batch count, which is not what "epsilon <= 1.0 per
    round" as a product requirement actually means.
    """

    def __init__(self, node_id: int, dataset, batch_size: int = 16,
                 lr: float = 1e-4, target_epsilon: float = 1.0,
                 target_delta: float = 1e-5,
                 max_grad_norm: float = 1.2, device: str = "cuda"):
        self.node_id = node_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.device = device

    def train_round(self, model: nn.Module, local_epochs: int = 1):
        # Must be in train() mode *before* make_private() -- Opacus validates
        # this on the incoming module, and the global model arrives here in
        # eval() mode after the previous round's evaluate_model() call.
        model = model.to(self.device).train()
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        sample_rate = self.batch_size / len(self.dataset)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            sample_rate=sample_rate,
            epochs=local_epochs,
        )

        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        model.train()
        total_loss, n_batches = 0.0, 0
        for _ in range(local_epochs):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        epsilon = privacy_engine.get_epsilon(delta=self.target_delta)
        # Opacus wraps the model in GradSampleModule; unwrap before returning state_dict
        raw_state = model._module.state_dict() if hasattr(model, "_module") else model.state_dict()
        state_dict: OrderedDict = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in raw_state.items()
        )
        avg_loss = total_loss / max(n_batches, 1)
        # noise_multiplier/sample_rate/n_batches let the caller track *cumulative*
        # privacy spend across rounds (see privacy_accounting.py) -- this round's
        # `epsilon` above only answers "how much did this one round cost in
        # isolation", not "how much has this node spent in total so far".
        return state_dict, len(self.dataset), epsilon, avg_loss, noise_multiplier, sample_rate, n_batches

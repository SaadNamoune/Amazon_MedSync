from collections import OrderedDict

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader


class LocalClient:
    """One simulated hospital node: trains the global model on its own
    private shard with DP-SGD (Opacus) so raw gradients never leave the
    node in a form that could be reverse-engineered to patient data."""

    def __init__(self, node_id: int, dataset, batch_size: int = 16,
                 lr: float = 1e-4, noise_multiplier: float = 1.0,
                 max_grad_norm: float = 1.2, device: str = "cuda"):
        self.node_id = node_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.device = device

    def train_round(self, model: nn.Module, local_epochs: int = 1):
        model = model.to(self.device)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=self.noise_multiplier,
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

        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        # Opacus wraps the model in GradSampleModule; unwrap before returning state_dict
        raw_state = model._module.state_dict() if hasattr(model, "_module") else model.state_dict()
        state_dict: OrderedDict = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in raw_state.items()
        )
        avg_loss = total_loss / max(n_batches, 1)
        return state_dict, len(self.dataset), epsilon, avg_loss

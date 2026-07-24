import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(model, dataset, device: str = "cuda", batch_size: int = 32):
    """Macro-averaged ROC-AUC across all 15 labels on a held-out set.
    Labels with only one class present in the eval batch are skipped
    (roc_auc_score is undefined for them) and reported separately."""
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    per_label_auc = {}
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if len(np.unique(col)) < 2:
            continue
        per_label_auc[i] = roc_auc_score(col, y_prob[:, i])

    macro_auc = float(np.mean(list(per_label_auc.values()))) if per_label_auc else float("nan")
    return macro_auc, per_label_auc

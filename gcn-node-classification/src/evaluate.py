"""
evaluate.py — Evaluation helpers for the GCN model.

Computes loss and accuracy on arbitrary index sets (train / val / test).
"""

import torch
import torch.nn.functional as F
from src.utils import accuracy, get_best_device, prepare_graph_tensors


def evaluate(
    model,
    features: torch.Tensor,
    adj: torch.sparse.FloatTensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    set_name: str = "Test",
    device: str = None,
):
    """
    Evaluate the model on the node subset given by `idx`.

    Parameters
    ----------
    model     : trained GCN model
    features  : (N, F) node feature matrix
    adj       : (N, N) normalised adjacency (sparse)
    labels    : (N,) ground-truth labels
    idx       : node indices to evaluate on
    set_name  : label used for printing (e.g. "Test", "Validation")

    Returns
    -------
    loss : float
    acc  : float
    """
    target_device = get_best_device(device)
    model = model.to(target_device)
    features, adj, labels, idx, _, _ = prepare_graph_tensors(
        features=features,
        adj=adj,
        labels=labels,
        idx_train=idx,
        device=target_device,
    )

    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss = F.nll_loss(output[idx], labels[idx]).item()
        acc = accuracy(output[idx], labels[idx])

    print(
        f"{set_name:>12s}  →  loss = {loss:.4f}  |  "
        f"accuracy = {acc:.4f}  |  device = {target_device}"
    )
    return loss, acc

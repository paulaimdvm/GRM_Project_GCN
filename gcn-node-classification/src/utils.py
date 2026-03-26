"""
utils.py — Miscellaneous helper functions.

    - accuracy():       compute classification accuracy from log-probabilities
    - set_seed():       reproducible experiments
    - print_config():   nicely print hyper-parameters
"""

import random
from typing import Optional

import numpy as np
import torch


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    output : (N, C) log-probabilities (output of the model)
    labels : (N,)   true labels

    Returns
    -------
    acc : float in [0, 1]
    """
    preds = output.argmax(dim=1)
    correct = preds.eq(labels).sum().item()
    return correct / len(labels)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_device(device: Optional[str] = None) -> torch.device:
    """
    Resolve the execution device.

    Priority when *device* is None: MPS -> CUDA -> CPU.
    If *device* is provided, validates availability and gracefully falls back to CPU.
    """
    if device is None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def prepare_graph_tensors(
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    idx_train: Optional[torch.Tensor] = None,
    idx_val: Optional[torch.Tensor] = None,
    idx_test: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dense_adj_on_mps: bool = True,
):
    """
    Move graph tensors to the target device and apply MPS compatibility fixes.

    MPS may not support sparse COO ops consistently across PyTorch versions.
    When *dense_adj_on_mps* is True, sparse adjacency is converted to dense on MPS.
    """
    if device is None:
        device = get_best_device()

    features = features.to(device)
    labels = labels.to(device) if labels is not None else None
    idx_train = idx_train.to(device) if idx_train is not None else None
    idx_val = idx_val.to(device) if idx_val is not None else None
    idx_test = idx_test.to(device) if idx_test is not None else None

    if device.type == "mps" and adj.is_sparse and dense_adj_on_mps:
        adj = adj.to_dense().to(device)
    else:
        adj = adj.to(device)

    return features, adj, labels, idx_train, idx_val, idx_test


def print_config(config: dict):
    """Pretty-print a configuration dictionary."""
    print("\n" + "=" * 50)
    print("  Hyper-parameters")
    print("=" * 50)
    for k, v in config.items():
        print(f"  {k:<20s} : {v}")
    print("=" * 50 + "\n")

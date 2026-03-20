"""
utils.py — Miscellaneous helper functions.

    - accuracy():       compute classification accuracy from log-probabilities
    - set_seed():       reproducible experiments
    - print_config():   nicely print hyper-parameters
"""

import random
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


def print_config(config: dict):
    """Pretty-print a configuration dictionary."""
    print("\n" + "=" * 50)
    print("  Hyper-parameters")
    print("=" * 50)
    for k, v in config.items():
        print(f"  {k:<20s} : {v}")
    print("=" * 50 + "\n")

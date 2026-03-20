"""
experiments.py -- Hyperparameter analysis and over-smoothing study.

Provides functions to systematically evaluate the impact of:
    - hidden layer size
    - dropout rate
    - learning rate
    - network depth (over-smoothing)

Each experiment trains a fresh model, evaluates on all splits, and
returns structured results suitable for plotting.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils import set_seed, accuracy
from src.model import GCN, DeepGCN
from src.train import train_model


# ---------------------------------------------------------------------------
#  Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    features: torch.Tensor,
    adj: torch.sparse.FloatTensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
    hidden: int = 16,
    dropout: float = 0.5,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    n_layers: int = 2,
    seed: int = 42,
):
    """
    Train a fresh GCN with the given hyperparameters and return accuracies.

    Parameters
    ----------
    features, adj, labels : dataset tensors.
    idx_train, idx_val, idx_test : split indices.
    hidden      : hidden dimension.
    dropout     : dropout probability.
    lr          : learning rate.
    weight_decay: L2 regularisation.
    epochs      : training epochs.
    n_layers    : number of GCN layers (2 = standard, >2 = deep).
    seed        : random seed.

    Returns
    -------
    train_acc, val_acc, test_acc : float
    history : dict
    """
    set_seed(seed)

    n_features = features.shape[1]
    n_classes = labels.max().item() + 1

    if n_layers == 2:
        model = GCN(n_features, n_hidden=hidden, n_classes=n_classes, dropout=dropout)
    else:
        model = DeepGCN(
            n_features, n_hidden=hidden, n_classes=n_classes,
            n_layers=n_layers, dropout=dropout,
        )

    history, _ = train_model(
        model, features, adj, labels,
        idx_train, idx_val,
        lr=lr, weight_decay=weight_decay, epochs=epochs,
        verbose=False,
    )

    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        train_acc = accuracy(output[idx_train], labels[idx_train])
        val_acc = accuracy(output[idx_val], labels[idx_val])
        test_acc = accuracy(output[idx_test], labels[idx_test])

    return train_acc, val_acc, test_acc, history


# ---------------------------------------------------------------------------
#  Sweep: hidden dimension
# ---------------------------------------------------------------------------

def sweep_hidden_size(
    features, adj, labels, idx_train, idx_val, idx_test,
    hidden_sizes=(4, 8, 16, 32, 64, 128),
    **kwargs,
):
    """
    Evaluate test accuracy across different hidden-layer sizes.

    Returns
    -------
    results : list of (hidden, train_acc, val_acc, test_acc)
    fig     : matplotlib.figure.Figure
    """
    results = []
    print("Sweep: Hidden Dimension")
    print("-" * 60)

    for h in hidden_sizes:
        tr, va, te, _ = run_single_experiment(
            features, adj, labels, idx_train, idx_val, idx_test,
            hidden=h, **kwargs,
        )
        results.append((h, tr, va, te))
        print(f"  hidden={h:<4d}  |  train={tr:.4f}  val={va:.4f}  test={te:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r[0] for r in results], [r[3] for r in results],
            "o-", label="Test", linewidth=2)
    ax.plot([r[0] for r in results], [r[2] for r in results],
            "s--", label="Validation", linewidth=2)
    ax.set_xlabel("Hidden Dimension")
    ax.set_ylabel("Accuracy")
    ax.set_title("Effect of Hidden Dimension")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return results, fig


# ---------------------------------------------------------------------------
#  Sweep: dropout rate
# ---------------------------------------------------------------------------

def sweep_dropout(
    features, adj, labels, idx_train, idx_val, idx_test,
    dropout_rates=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9),
    **kwargs,
):
    """
    Evaluate test accuracy across different dropout rates.

    Returns
    -------
    results : list of (dropout, train_acc, val_acc, test_acc)
    fig     : matplotlib.figure.Figure
    """
    results = []
    print("Sweep: Dropout Rate")
    print("-" * 60)

    for d in dropout_rates:
        tr, va, te, _ = run_single_experiment(
            features, adj, labels, idx_train, idx_val, idx_test,
            dropout=d, **kwargs,
        )
        results.append((d, tr, va, te))
        print(f"  dropout={d:.1f}  |  train={tr:.4f}  val={va:.4f}  test={te:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r[0] for r in results], [r[3] for r in results],
            "o-", label="Test", linewidth=2)
    ax.plot([r[0] for r in results], [r[2] for r in results],
            "s--", label="Validation", linewidth=2)
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Effect of Dropout")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return results, fig


# ---------------------------------------------------------------------------
#  Sweep: learning rate
# ---------------------------------------------------------------------------

def sweep_learning_rate(
    features, adj, labels, idx_train, idx_val, idx_test,
    learning_rates=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
    **kwargs,
):
    """
    Evaluate test accuracy across different learning rates.

    Returns
    -------
    results : list of (lr, train_acc, val_acc, test_acc)
    fig     : matplotlib.figure.Figure
    """
    results = []
    print("Sweep: Learning Rate")
    print("-" * 60)

    for lr_val in learning_rates:
        tr, va, te, _ = run_single_experiment(
            features, adj, labels, idx_train, idx_val, idx_test,
            lr=lr_val, **kwargs,
        )
        results.append((lr_val, tr, va, te))
        print(f"  lr={lr_val:<8.4f}  |  train={tr:.4f}  val={va:.4f}  test={te:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(
        [r[0] for r in results], [r[3] for r in results],
        "o-", label="Test", linewidth=2,
    )
    ax.semilogx(
        [r[0] for r in results], [r[2] for r in results],
        "s--", label="Validation", linewidth=2,
    )
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Effect of Learning Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return results, fig


# ---------------------------------------------------------------------------
#  Over-smoothing study (variable depth)
# ---------------------------------------------------------------------------

def sweep_depth(
    features, adj, labels, idx_train, idx_val, idx_test,
    layer_counts=(2, 3, 4, 6, 8, 12),
    hidden: int = 16,
    **kwargs,
):
    """
    Evaluate the over-smoothing effect by varying network depth.

    As the number of GCN layers increases, repeated neighbourhood
    aggregation causes node embeddings to converge, destroying the
    discriminative signal.  This experiment quantifies that effect.

    Returns
    -------
    results : list of (n_layers, train_acc, val_acc, test_acc)
    fig     : matplotlib.figure.Figure
    """
    results = []
    print("Sweep: Network Depth (Over-Smoothing Study)")
    print("-" * 60)

    for n_layers in layer_counts:
        tr, va, te, _ = run_single_experiment(
            features, adj, labels, idx_train, idx_val, idx_test,
            hidden=hidden, n_layers=n_layers, **kwargs,
        )
        results.append((n_layers, tr, va, te))
        print(f"  layers={n_layers:<3d}  |  train={tr:.4f}  val={va:.4f}  test={te:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r[0] for r in results], [r[3] for r in results],
            "o-", label="Test", linewidth=2, color="tab:red")
    ax.plot([r[0] for r in results], [r[2] for r in results],
            "s--", label="Validation", linewidth=2, color="tab:orange")
    ax.plot([r[0] for r in results], [r[1] for r in results],
            "^:", label="Train", linewidth=2, color="tab:blue")
    ax.set_xlabel("Number of GCN Layers")
    ax.set_ylabel("Accuracy")
    ax.set_title("Over-Smoothing: Accuracy vs. Network Depth")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return results, fig

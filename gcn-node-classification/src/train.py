"""
train.py -- Training loop for the GCN model.

Trains the model using negative log-likelihood (cross-entropy) loss on the
labelled training nodes with Adam optimiser.  Tracks training + validation
metrics every epoch and optionally captures hidden-layer embeddings at
user-specified epochs for later visualisation (e.g., t-SNE).
"""

import copy
import time
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

try:
    from src.utils import accuracy, get_best_device, prepare_graph_tensors
except ModuleNotFoundError:
    # Allow direct execution via: python src/train.py
    from utils import accuracy, get_best_device, prepare_graph_tensors


def train_model(
    model,
    features: torch.Tensor,
    adj: torch.sparse.FloatTensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    verbose: bool = True,
    snapshot_epochs: Optional[list] = None,
    device: Optional[str] = None,
):
    """
    Train the GCN model and return training history.

    Parameters
    ----------
    model           : nn.Module  -- GCN or DeepGCN model
    features        : (N, F)     -- node features
    adj             : (N, N)     -- normalised adjacency (sparse)
    labels          : (N,)       -- ground-truth labels
    idx_train       : training node indices
    idx_val         : validation node indices
    lr              : learning rate (default 0.01)
    weight_decay    : L2 regularisation (default 5e-4)
    epochs          : number of training epochs (default 200)
    verbose         : print progress every 10 epochs
    snapshot_epochs : list of ints, optional
        Epoch numbers at which to capture a copy of the model state for
        later embedding extraction.  For example [0, 10, 50, 100, 200].
        Epoch 0 captures the state *before* any gradient update.

    Returns
    -------
    history : dict
        Keys: 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
    snapshots : dict[int, dict]
        Mapping from epoch number to a deep-copied ``state_dict``.
        Only populated when *snapshot_epochs* is not None.
    """
    if snapshot_epochs is None:
        snapshot_epochs = []

    target_device = get_best_device(device)
    model = model.to(target_device)
    features, adj, labels, idx_train, idx_val, _ = prepare_graph_tensors(
        features=features,
        adj=adj,
        labels=labels,
        idx_train=idx_train,
        idx_val=idx_val,
        device=target_device,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    snapshots = {}

    # Capture the untrained state (epoch 0) if requested
    if 0 in snapshot_epochs:
        snapshots[0] = copy.deepcopy(model.state_dict())

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # -- Training step
        model.train()
        optimizer.zero_grad()

        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        # -- Validation step
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

        # -- Record history
        history["train_loss"].append(loss_train.item())
        history["train_acc"].append(acc_train)
        history["val_loss"].append(loss_val.item())
        history["val_acc"].append(acc_val)

        # -- Snapshot for visualisation
        if epoch in snapshot_epochs:
            snapshots[epoch] = copy.deepcopy(model.state_dict())

        # -- Print progress
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:>3d}/{epochs}  |  "
                f"train_loss={loss_train.item():.4f}  train_acc={acc_train:.4f}  |  "
                f"val_loss={loss_val.item():.4f}  val_acc={acc_val:.4f}"
            )

    elapsed = time.time() - t_start
    if verbose:
        print(f"Device: {target_device}")
        print(f"\nTraining completed in {elapsed:.2f}s")

    return history, snapshots

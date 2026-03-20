"""
dataset.py — Load and prepare the Cora citation dataset.

The Cora dataset consists of 2708 machine-learning papers classified into
7 classes.  Each paper is described by a binary bag-of-words vector of
1433 dimensions.  The citation graph has 5429 edges.

This module provides:
    - load_cora()   → features, labels, adjacency matrix, and train/val/test masks
"""

import os
import numpy as np
import scipy.sparse as sp
import torch


# ─── Label encoding ──────────────────────────────────────────────────────────

CLASSES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]


def _encode_labels(raw_labels: list[str]) -> np.ndarray:
    """Map string class names to integer indices."""
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    return np.array([class_to_idx[l] for l in raw_labels], dtype=np.int64)


# ─── Main loader ─────────────────────────────────────────────────────────────

def load_cora(data_dir: str = None):
    """
    Load the Cora dataset from `data_dir`.

    Parameters
    ----------
    data_dir : str
        Path to the folder that contains `cora.content` and `cora.cites`.
        If None, defaults to ``../cora/cora/`` relative to this file.

    Returns
    -------
    features   : torch.FloatTensor   (N, 1433) — node feature matrix
    labels     : torch.LongTensor    (N,)      — class labels [0..6]
    adj        : scipy.sparse.coo    (N, N)    — adjacency matrix (no self-loops)
    idx_train  : torch.LongTensor              — training node indices
    idx_val    : torch.LongTensor              — validation node indices
    idx_test   : torch.LongTensor              — test node indices
    """

    # ── Resolve data directory ────────────────────────────────────────────
    if data_dir is None:
        # Default: <project>/data  or  <workspace>/cora/cora
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir_candidate = os.path.join(project_root, "data")
        if os.path.exists(os.path.join(data_dir_candidate, "cora.content")):
            data_dir = data_dir_candidate
        else:
            # Try the workspace-level cora/cora folder
            workspace_root = os.path.dirname(project_root)
            data_dir = os.path.join(workspace_root, "cora", "cora")

    content_path = os.path.join(data_dir, "cora.content")
    cites_path = os.path.join(data_dir, "cora.cites")

    assert os.path.isfile(content_path), f"File not found: {content_path}"
    assert os.path.isfile(cites_path), f"File not found: {cites_path}"

    # ── Parse cora.content ────────────────────────────────────────────────
    # Format: <paper_id> <1433 binary features> <class_label>
    paper_ids = []
    features_list = []
    raw_labels = []

    with open(content_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            paper_ids.append(parts[0])
            features_list.append([int(x) for x in parts[1:-1]])
            raw_labels.append(parts[-1])

    # Build a mapping from paper_id → contiguous index
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    num_nodes = len(paper_ids)

    features = np.array(features_list, dtype=np.float32)
    labels = _encode_labels(raw_labels)

    # ── Parse cora.cites and build adjacency matrix ───────────────────────
    # Format: <cited_paper_id>  <citing_paper_id>
    edges = []
    with open(cites_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            src, dst = parts
            if src in id_to_idx and dst in id_to_idx:
                i = id_to_idx[src]
                j = id_to_idx[dst]
                edges.append((i, j))
                edges.append((j, i))          # undirected

    rows, cols = zip(*edges)
    adj = sp.coo_matrix(
        (np.ones(len(rows)), (np.array(rows), np.array(cols))),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    # Remove duplicate entries
    adj = adj.tocsr()
    adj[adj > 1] = 1
    adj = adj.tocoo()

    # ── Row-normalise features ────────────────────────────────────────────
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1  # avoid division by zero
    features = features / row_sum

    # ── Train / validation / test split (Kipf convention) ─────────────────
    # train: first 140,  val: 140–640,  test: 1708–2708
    idx_train = torch.arange(140, dtype=torch.long)
    idx_val = torch.arange(200, 500, dtype=torch.long)
    idx_test = torch.arange(500, 1500, dtype=torch.long)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    return features, labels, adj, idx_train, idx_val, idx_test

"""
preprocess.py — Graph pre-processing utilities.

Implements the symmetric normalisation of the adjacency matrix used in
the GCN paper (Kipf & Welling, 2017):

    Â_norm = D̃^{-1/2}  Ã  D̃^{-1/2}

where  Ã = A + I_N  (adjacency with self-loops)
and    D̃_ii = Σ_j Ã_ij  (degree matrix of Ã).
"""

import numpy as np
import scipy.sparse as sp
import torch


def add_self_loops(adj: sp.spmatrix) -> sp.coo_matrix:
    """
    Add self-loops to the adjacency matrix:  Ã = A + I_N.

    Parameters
    ----------
    adj : scipy.sparse matrix (N, N)

    Returns
    -------
    adj_tilde : scipy.sparse.coo_matrix (N, N)
    """
    N = adj.shape[0]
    adj_tilde = adj + sp.eye(N, dtype=np.float32)
    return adj_tilde.tocoo()


def compute_normalized_adjacency(adj: sp.spmatrix) -> torch.FloatTensor:
    """
    Compute the symmetrically normalised adjacency matrix:

        D̃^{-1/2}  Ã  D̃^{-1/2}

    where Ã already contains self-loops.

    Parameters
    ----------
    adj : scipy.sparse matrix (N, N)  —  should already include self-loops.

    Returns
    -------
    adj_norm : torch.sparse.FloatTensor (N, N)
    """
    adj = sp.coo_matrix(adj, dtype=np.float32)

    # Degree vector of Ã
    degree = np.array(adj.sum(axis=1)).flatten()  # shape (N,)

    # D̃^{-1/2}
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    # Build sparse diagonal D̃^{-1/2}
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    # Symmetric normalisation:  D̃^{-1/2} Ã D̃^{-1/2}
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    adj_norm = adj_norm.tocoo()

    # Convert to PyTorch sparse tensor
    indices = torch.LongTensor(
        np.vstack([adj_norm.row, adj_norm.col])
    )
    values = torch.FloatTensor(adj_norm.data)
    shape = torch.Size(adj_norm.shape)

    return torch.sparse_coo_tensor(indices, values, shape).float()


def preprocess_adjacency(adj: sp.spmatrix) -> torch.sparse.FloatTensor:
    """
    Full preprocessing pipeline:
        1.  Add self-loops   →  Ã = A + I
        2.  Normalise        →  D̃^{-1/2} Ã D̃^{-1/2}

    Parameters
    ----------
    adj : scipy.sparse matrix (N, N)

    Returns
    -------
    adj_norm : torch.sparse.FloatTensor (N, N)
    """
    adj_tilde = add_self_loops(adj)
    return compute_normalized_adjacency(adj_tilde)

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


def compute_scaled_laplacian(adj: sp.spmatrix) -> torch.sparse.FloatTensor:
    """
    Compute the rescaled Laplacian used by the Chebyshev polynomial filter:

        L̃ = (2 / λ_max) · L_norm  −  I_N

    where  L_norm = I_N − D̃^{-1/2} Ã D̃^{-1/2}  is the symmetric normalised
    graph Laplacian (computed from Ã = A + I), and λ_max is its largest
    eigenvalue (approximated as 2 following Kipf & Welling, Eq.4).

    Parameters
    ----------
    adj : scipy.sparse matrix (N, N)  —  *raw* adjacency (without self-loops).

    Returns
    -------
    L_tilde : torch.sparse.FloatTensor (N, N)
    """
    # Ã = A + I  (add self-loops)
    adj_tilde = add_self_loops(adj)
    adj_coo = sp.coo_matrix(adj_tilde, dtype=np.float32)
    N = adj_coo.shape[0]

    # D̃^{-1/2}
    degree = np.array(adj_coo.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    # L_norm = I - D̃^{-1/2} Ã D̃^{-1/2}
    L_norm = sp.eye(N, dtype=np.float32) - D_inv_sqrt @ adj_coo @ D_inv_sqrt

    # Approximate λ_max ≈ 2 (as stated in the paper, Section 2.1)
    lambda_max = 2.0

    # L̃ = (2 / λ_max) · L_norm  −  I  =  L_norm − I   (when λ_max = 2)
    L_tilde = (2.0 / lambda_max) * L_norm - sp.eye(N, dtype=np.float32)
    L_tilde = sp.coo_matrix(L_tilde)

    # Convert to PyTorch sparse tensor
    indices = torch.LongTensor(np.vstack([L_tilde.row, L_tilde.col]))
    values = torch.FloatTensor(L_tilde.data)
    shape = torch.Size(L_tilde.shape)

    return torch.sparse_coo_tensor(indices, values, shape).float()


def preprocess_chebyshev(adj: sp.spmatrix):
    """
    Full preprocessing pipeline for the Chebyshev K-hop model:
        1.  adj_norm  = D̃^{-1/2} Ã D̃^{-1/2}   (for the standard GCN)
        2.  L_tilde   = (2/λ_max) L_norm − I      (for the Chebyshev filter)

    Parameters
    ----------
    adj : scipy.sparse matrix (N, N)  —  raw adjacency.

    Returns
    -------
    adj_norm : torch.sparse.FloatTensor (N, N)
    L_tilde  : torch.sparse.FloatTensor (N, N)
    """
    adj_norm = preprocess_adjacency(adj)
    L_tilde = compute_scaled_laplacian(adj)
    return adj_norm, L_tilde

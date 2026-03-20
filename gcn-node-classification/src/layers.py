"""
layers.py — Graph Convolution layer (manual implementation).

Implements a single GCN layer following the propagation rule from
Kipf & Welling (2017):

    H^{(l+1)} = σ( D̃^{-1/2} Ã D̃^{-1/2} · H^{(l)} · W^{(l)} )

No external GNN library is used; the sparse matrix multiplication is
performed with standard PyTorch operations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    A single Graph Convolution layer.

    Parameters
    ----------
    in_features  : int — dimension of input node features
    out_features : int — dimension of output node features
    bias         : bool — whether to add a learnable bias (default True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable weight matrix  W^{(l)}  of shape (in, out)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier uniform initialisation (as in the original code)."""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Forward pass:

            output = Â_norm · X · W  (+  bias)

        Parameters
        ----------
        x   : torch.Tensor          (N, in_features)  — node features H^{(l)}
        adj : torch.sparse.Tensor   (N, N)            — normalised adjacency

        Returns
        -------
        out : torch.Tensor          (N, out_features)
        """
        # Step 1: Linear transformation  X · W
        support = torch.mm(x, self.weight)          # (N, out_features)

        # Step 2: Neighbourhood aggregation  Â · (X W)
        output = torch.spmm(adj, support)           # sparse × dense  → (N, out_features)

        # Step 3: Add bias
        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.in_features} → {self.out_features})"
        )

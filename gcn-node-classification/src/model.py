"""
model.py -- Graph Convolutional Network for node classification.

Provides two model variants:

1.  ``GCN``     -- The standard 2-layer GCN used by Kipf & Welling (2017).
2.  ``DeepGCN`` -- A variable-depth GCN for studying **over-smoothing**.

Architecture (2-layer, default)
-------------------------------
    Input  (N, n_features)
      |
      +-- GraphConvolution  ->  (N, n_hidden)
      +-- ReLU
      +-- Dropout
      |
      +-- GraphConvolution  ->  (N, n_classes)
      +-- Log-Softmax       ->  (N, n_classes)

Reference
---------
    Kipf, T.N. and Welling, M., 2017.
    Semi-Supervised Classification with Graph Convolutional Networks.
    ICLR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import GraphConvolution


class GCN(nn.Module):
    """
    Two-layer GCN for semi-supervised node classification.

    Parameters
    ----------
    n_features : int   -- input feature dimension (e.g. 1433 for Cora)
    n_hidden   : int   -- hidden layer dimension (default 16)
    n_classes  : int   -- number of output classes (default 7)
    dropout    : float -- dropout rate (default 0.5)
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 16,
        n_classes: int = 7,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.gc1 = GraphConvolution(n_features, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x   : (N, n_features)  -- node feature matrix
        adj : (N, N)           -- normalised adjacency (sparse)

        Returns
        -------
        log_probs : (N, n_classes)
        """
        # First GCN layer  +  ReLU  +  Dropout
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Second GCN layer  +  log-softmax
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Return the hidden-layer activations (after ReLU, before dropout).

        These embeddings live in R^{n_hidden} and are suitable for
        dimensionality-reduction visualisations (t-SNE, UMAP).

        Parameters
        ----------
        x   : (N, n_features)
        adj : (N, N)  -- normalised adjacency (sparse)

        Returns
        -------
        embeddings : (N, n_hidden)
        """
        x = self.gc1(x, adj)
        x = F.relu(x)
        return x


class DeepGCN(nn.Module):
    """
    Variable-depth GCN for studying the **over-smoothing** phenomenon.

    As the number of layers increases, node representations tend to converge
    to the same vector, collapsing the discriminative power of the model.
    This class allows creating GCNs with an arbitrary number of layers to
    reproduce and visualise this effect.

    Parameters
    ----------
    n_features : int   -- input feature dimension
    n_hidden   : int   -- hidden layer dimension
    n_classes  : int   -- output classes
    n_layers   : int   -- total number of GCN layers (>= 2)
    dropout    : float -- dropout probability
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 16,
        n_classes: int = 7,
        n_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        assert n_layers >= 2, "DeepGCN requires at least 2 layers."

        self.dropout = dropout
        self.layers = nn.ModuleList()

        # First layer:  n_features -> n_hidden
        self.layers.append(GraphConvolution(n_features, n_hidden))

        # Intermediate hidden layers:  n_hidden -> n_hidden
        for _ in range(n_layers - 2):
            self.layers.append(GraphConvolution(n_hidden, n_hidden))

        # Final layer:  n_hidden -> n_classes
        self.layers.append(GraphConvolution(n_hidden, n_classes))

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        # Output layer (no ReLU, apply log-softmax)
        x = self.layers[-1](x, adj)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Return the activations after the penultimate layer.

        For a network with L layers, this returns the output of layer L-1
        (after ReLU, before dropout), which is the last hidden
        representation before the classification head.
        """
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = F.relu(x)
        return x

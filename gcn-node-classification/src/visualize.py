"""
visualize.py -- Visualisation utilities for GCN analysis.

Provides:
    - ``plot_training_curves``    : loss / accuracy over epochs.
    - ``plot_confusion_matrix``   : confusion matrix on any split.
    - ``compute_tsne``            : t-SNE projection of node embeddings.
    - ``plot_tsne``               : single t-SNE scatter plot.
    - ``plot_embedding_evolution``: multi-panel t-SNE across training stages.
    - ``extract_subgraph``        : extract a readable k-hop neighbourhood.
    - ``animate_graph_evolution`` : animated graph where node positions are
                                   t-SNE coordinates, using celluloid.Camera.

All plotting functions return the matplotlib ``Figure`` object so that
callers can further customise or save the figure.
"""

import copy
import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from celluloid import Camera
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from src.dataset import CLASSES
from src.utils import accuracy


# ---------------------------------------------------------------------------
#  Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict) -> plt.Figure:
    """
    Plot training and validation loss / accuracy curves.

    Parameters
    ----------
    history : dict
        Must contain keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Loss
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train", linewidth=1.5)
    ax.plot(history["val_loss"], label="Validation", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log-Likelihood")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # -- Accuracy
    ax = axes[1]
    ax.plot(history["train_acc"], label="Train", linewidth=1.5)
    ax.plot(history["val_acc"], label="Validation", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Training Curves", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
#  Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    model,
    features: torch.Tensor,
    adj: torch.sparse.FloatTensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    set_name: str = "Test",
) -> plt.Figure:
    """
    Compute and display a confusion matrix for the given node subset.

    Parameters
    ----------
    model    : trained GCN model
    features : (N, F)
    adj      : (N, N) normalised adjacency (sparse)
    labels   : (N,) ground-truth labels
    idx      : node indices to evaluate
    set_name : label for the plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        preds = output.argmax(dim=1)

    y_true = labels[idx].numpy()
    y_pred = preds[idx].numpy()

    n_cls = len(CLASSES)
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({set_name} Set)")

    for i in range(n_cls):
        for j in range(n_cls):
            colour = "white" if conf[i, j] > conf.max() / 2 else "black"
            ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                    color=colour, fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
#  t-SNE utilities
# ---------------------------------------------------------------------------

def compute_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute 2-D t-SNE projection of high-dimensional embeddings.

    Parameters
    ----------
    embeddings   : (N, D) array of node embeddings.
    perplexity   : t-SNE perplexity (controls the balance between local
                   and global structure).
    random_state : seed for reproducibility.

    Returns
    -------
    coords : (N, 2) array of 2-D coordinates.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE of GCN Embeddings",
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Scatter plot of 2-D t-SNE coordinates coloured by class label.

    Parameters
    ----------
    coords : (N, 2)  -- t-SNE coordinates.
    labels : (N,)    -- integer class labels.
    title  : str     -- plot title.
    ax     : matplotlib Axes, optional.  Created if None.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    n_classes = len(CLASSES)
    cmap = plt.cm.get_cmap("tab10", n_classes)

    for c in range(n_classes):
        mask = labels == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(c)], label=CLASSES[c],
            s=12, alpha=0.7, edgecolors="none",
        )

    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=2, loc="best", framealpha=0.8)
    return ax


# ---------------------------------------------------------------------------
#  Embedding evolution across training
# ---------------------------------------------------------------------------

def extract_embeddings(model, features, adj):
    """
    Extract hidden-layer embeddings from a model.

    Parameters
    ----------
    model    : GCN or DeepGCN with a ``get_embeddings`` method.
    features : (N, F)
    adj      : (N, N) normalised adjacency (sparse)

    Returns
    -------
    embeddings : np.ndarray  (N, n_hidden)
    """
    model.eval()
    with torch.no_grad():
        emb = model.get_embeddings(features, adj)
    return emb.cpu().numpy()


def plot_embedding_evolution(
    model,
    snapshots: dict,
    features: torch.Tensor,
    adj: torch.sparse.FloatTensor,
    labels: torch.Tensor,
    perplexity: float = 30.0,
    save_path: str = None,
) -> plt.Figure:
    """
    Generate a multi-panel figure showing t-SNE projections of the latent
    space at different training stages.

    For each epoch in *snapshots*, the model weights are temporarily loaded,
    hidden-layer embeddings are extracted, and a t-SNE projection is computed.

    Parameters
    ----------
    model       : GCN or DeepGCN instance (architecture must match snapshots).
    snapshots   : dict[int, state_dict] -- output of ``train_model``.
    features    : (N, F)
    adj         : (N, N) sparse normalised adjacency.
    labels      : (N,) ground-truth labels (used only for colouring).
    perplexity  : t-SNE perplexity.
    save_path   : str, optional.  If provided, save the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    sorted_epochs = sorted(snapshots.keys())
    n_panels = len(sorted_epochs)

    if n_panels == 0:
        print("No snapshots available. Skipping embedding evolution plot.")
        return None

    # Determine grid layout: aim for roughly 3 columns
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 5.5 * n_rows),
        squeeze=False,
    )

    labels_np = labels.numpy()

    # Save current state so we can restore it afterwards
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    for idx, epoch in enumerate(sorted_epochs):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        # Load snapshot weights
        model.load_state_dict(snapshots[epoch])

        # Extract embeddings and project with t-SNE
        emb = extract_embeddings(model, features, adj)
        coords = compute_tsne(emb, perplexity=perplexity)

        title = f"Epoch {epoch}" if epoch > 0 else "Epoch 0 (untrained)"
        plot_tsne(coords, labels_np, title=title, ax=ax)

    # Hide unused axes
    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    # Restore original model weights
    model.load_state_dict(original_state)

    fig.suptitle(
        "Evolution of Latent Space During Training (t-SNE)",
        fontsize=15, y=1.01,
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
#  Subgraph extraction
# ---------------------------------------------------------------------------

def extract_subgraph(
    adj: sp.spmatrix,
    seed_node: int = None,
    hops: int = 2,
    max_nodes: int = 150,
    random_state: int = 42,
) -> np.ndarray:
    """
    Extract a small, readable subgraph from the full adjacency matrix.

    Starting from *seed_node*, performs a breadth-first expansion up to
    *hops* neighbours.  The result is capped at *max_nodes* to keep the
    plot legible.

    Parameters
    ----------
    adj          : scipy.sparse matrix (N, N)  -- original adjacency.
    seed_node    : int or None  -- starting node.  Chosen at random if None.
    hops         : int          -- neighbourhood radius (default 2).
    max_nodes    : int          -- hard cap on the number of returned nodes.
    random_state : int          -- seed for reproducibility when choosing the
                                   starting node.

    Returns
    -------
    subgraph_nodes : np.ndarray of int  -- global indices of selected nodes.
    """
    rng = random.Random(random_state)
    N = adj.shape[0]

    if seed_node is None:
        seed_node = rng.randint(0, N - 1)

    adj_csr = adj.tocsr()
    visited = {seed_node}
    frontier = {seed_node}

    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            neighbours = adj_csr[node].indices.tolist()
            next_frontier.update(neighbours)
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if len(visited) >= max_nodes:
            break

    # Cap to max_nodes, keeping seed in the set
    subgraph_nodes = sorted(visited)
    if len(subgraph_nodes) > max_nodes:
        subgraph_nodes.remove(seed_node)
        rng.shuffle(subgraph_nodes)
        subgraph_nodes = [seed_node] + subgraph_nodes[: max_nodes - 1]
        subgraph_nodes.sort()

    return np.array(subgraph_nodes, dtype=int)


def _build_networkx_subgraph(
    adj: sp.spmatrix,
    subgraph_nodes: np.ndarray,
) -> nx.Graph:
    """
    Build a NetworkX graph for the selected node subset.

    Parameters
    ----------
    adj             : scipy.sparse (N, N)  -- full adjacency.
    subgraph_nodes  : 1-D array of global node indices.

    Returns
    -------
    G : nx.Graph  -- undirected graph with nodes relabelled to 0..len-1.
    global_to_local : dict  -- maps global index -> local index.
    """
    global_to_local = {g: l for l, g in enumerate(subgraph_nodes)}
    adj_csr = adj.tocsr()

    G = nx.Graph()
    G.add_nodes_from(range(len(subgraph_nodes)))

    for g_idx in subgraph_nodes:
        local_src = global_to_local[g_idx]
        neighbours = adj_csr[g_idx].indices
        for g_neigh in neighbours:
            if g_neigh in global_to_local:
                local_dst = global_to_local[g_neigh]
                if local_src < local_dst:          # avoid duplicate edges
                    G.add_edge(local_src, local_dst)

    return G, global_to_local


# ---------------------------------------------------------------------------
#  Animated graph evolution (celluloid.Camera)
# ---------------------------------------------------------------------------

def animate_graph_evolution(
    model,
    features: torch.Tensor,
    adj_norm: torch.sparse.FloatTensor,
    adj_raw: sp.spmatrix,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    snapshot_step: int = 5,
    subgraph_nodes: np.ndarray = None,
    hops: int = 2,
    max_nodes: int = 150,
    perplexity: float = 30.0,
    save_path: str = None,
    seed: int = 42,
):
    """
    Train a GCN and produce an animation of the subgraph layout evolving
    as node embeddings change during training.

    At every *snapshot_step* epochs (and at epoch 0), the hidden-layer
    embeddings are projected to 2-D via t-SNE and used as node positions
    in a NetworkX graph drawing.  ``celluloid.Camera`` captures each frame
    and assembles them into a Matplotlib ``ArtistAnimation``.

    Parameters
    ----------
    model           : GCN or DeepGCN instance.
    features        : (N, F)  -- node feature matrix.
    adj_norm        : (N, N)  -- normalised adjacency (sparse torch tensor).
    adj_raw         : scipy.sparse (N, N) -- original adjacency used to
                      build the NetworkX subgraph (before self-loops).
    labels          : (N,) ground-truth labels.
    idx_train       : training indices (for accuracy reporting).
    idx_val         : validation indices (for accuracy reporting).
    lr, weight_decay, epochs : optimiser / training config.
    snapshot_step   : int  -- capture a frame every this many epochs.
    subgraph_nodes  : array of int, optional.  Pre-selected nodes.  If
                      None, a subgraph is extracted automatically via
                      ``extract_subgraph``.
    hops            : int  -- BFS radius for subgraph extraction.
    max_nodes       : int  -- cap on subgraph size.
    perplexity      : float -- t-SNE perplexity (auto-adjusted if the
                      subgraph is too small).
    save_path       : str  -- if provided, save the animation as a GIF.
    seed            : int  -- random seed.

    Returns
    -------
    anim : matplotlib.animation.ArtistAnimation
    """
    from src.utils import set_seed
    import torch.optim as optim

    set_seed(seed)

    # -- Extract or validate subgraph
    if subgraph_nodes is None:
        subgraph_nodes = extract_subgraph(
            adj_raw, seed_node=None, hops=hops,
            max_nodes=max_nodes, random_state=seed,
        )
    n_sub = len(subgraph_nodes)
    print(f"Subgraph: {n_sub} nodes selected.")

    # Build NetworkX graph for edge drawing
    G, global_to_local = _build_networkx_subgraph(adj_raw, subgraph_nodes)
    edge_list = list(G.edges())

    # Adjust perplexity if the subgraph is small
    effective_perplexity = min(perplexity, max(5.0, (n_sub - 1) / 3.0))

    # Ground-truth labels for the subgraph
    sub_labels = labels[subgraph_nodes].numpy()
    n_classes = len(CLASSES)
    cmap = plt.cm.get_cmap("tab10", n_classes)
    node_colours = [cmap(sub_labels[i]) for i in range(n_sub)]

    # -- Collect the epochs at which we snapshot
    snapshot_epochs = sorted(
        set([0] + list(range(snapshot_step, epochs + 1, snapshot_step)))
    )

    # -- Set up figure and camera
    fig, ax = plt.subplots(figsize=(8, 7))
    camera = Camera(fig)

    # -- Legend (static, drawn once per frame so Camera picks it up)
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=cmap(c),
            markersize=8, label=CLASSES[c],
        )
        for c in range(n_classes)
    ]

    # -- Training + animation loop
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _capture_frame(epoch, loss_val, acc_val):
        """Extract embeddings, project via t-SNE, draw graph."""
        model.eval()
        with torch.no_grad():
            emb_full = model.get_embeddings(features, adj_norm).cpu().numpy()
        emb_sub = emb_full[subgraph_nodes]

        coords = compute_tsne(emb_sub, perplexity=effective_perplexity)
        pos = {i: (coords[i, 0], coords[i, 1]) for i in range(n_sub)}

        # Draw edges
        for u, v in edge_list:
            x_coords = [pos[u][0], pos[v][0]]
            y_coords = [pos[u][1], pos[v][1]]
            ax.plot(x_coords, y_coords, color="grey", alpha=0.15,
                    linewidth=0.5, zorder=1)

        # Draw nodes
        xs = [pos[i][0] for i in range(n_sub)]
        ys = [pos[i][1] for i in range(n_sub)]
        ax.scatter(xs, ys, c=node_colours, s=30, zorder=2,
                   edgecolors="black", linewidths=0.3, alpha=0.85)

        # Title with epoch and metrics
        epoch_label = f"Epoch {epoch}" if epoch > 0 else "Epoch 0 (untrained)"
        title_text = f"{epoch_label}  |  val_loss={loss_val:.3f}  val_acc={acc_val:.3f}"
        ax.text(
            0.5, 1.02, title_text, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(handles=legend_handles, fontsize=7, loc="lower right",
                  framealpha=0.8, markerscale=1.0)

        camera.snap()

    # -- Frame at epoch 0 (untrained)
    model.eval()
    with torch.no_grad():
        output = model(features, adj_norm)
        loss_0 = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_0 = accuracy(output[idx_val], labels[idx_val])
    _capture_frame(0, loss_0, acc_0)

    # -- Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        output = model(features, adj_norm)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if epoch in snapshot_epochs:
            model.eval()
            with torch.no_grad():
                output = model(features, adj_norm)
                loss_v = F.nll_loss(output[idx_val], labels[idx_val]).item()
                acc_v = accuracy(output[idx_val], labels[idx_val])
            _capture_frame(epoch, loss_v, acc_v)

            if epoch % 50 == 0 or epoch == epochs:
                print(f"  Captured frame at epoch {epoch:>3d}  "
                      f"(val_loss={loss_v:.4f}, val_acc={acc_v:.4f})")

    # -- Build animation
    anim = camera.animate(interval=300, blit=False)

    if save_path is not None:
        anim.save(save_path, writer="pillow", dpi=120)
        print(f"Animation saved to: {save_path}")

    return anim

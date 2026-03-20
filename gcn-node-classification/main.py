"""
main.py — Entry point for the GCN node-classification project.

Usage
-----
    python main.py                            # use defaults
    python main.py --epochs 300 --lr 0.005    # override hyper-parameters
    python main.py --data_dir path/to/cora    # custom data path
"""

import argparse

from src.utils import set_seed, print_config
from src.dataset import load_cora
from src.preprocess import preprocess_adjacency
from src.model import GCN
from src.train import train_model
from src.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="GCN for semi-supervised node classification on Cora"
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the Cora data folder (cora.content + cora.cites)")
    parser.add_argument("--hidden", type=int, default=16,
                        help="Number of hidden units (default: 16)")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate (default: 0.5)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay / L2 regularisation (default: 5e-4)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────
    set_seed(args.seed)

    config = {
        "hidden": args.hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    print_config(config)

    # ── 1. Load dataset ──────────────────────────────────────────────────
    print("Loading Cora dataset …")
    features, labels, adj, idx_train, idx_val, idx_test = load_cora(args.data_dir)
    print(f"  Nodes: {features.shape[0]}  |  Features: {features.shape[1]}  |  "
          f"Edges: {adj.nnz // 2}")

    # ── 2. Preprocess adjacency matrix ───────────────────────────────────
    print("Preprocessing adjacency matrix (Â = D̃⁻¹/² Ã D̃⁻¹/²) …")
    adj_norm = preprocess_adjacency(adj)

    # ── 3. Build model ───────────────────────────────────────────────────
    n_features = features.shape[1]
    n_classes = labels.max().item() + 1
    model = GCN(
        n_features=n_features,
        n_hidden=args.hidden,
        n_classes=n_classes,
        dropout=args.dropout,
    )
    print(f"\nModel architecture:\n{model}\n")

    # ── 4. Train ─────────────────────────────────────────────────────────
    history, _ = train_model(
        model, features, adj_norm, labels,
        idx_train, idx_val,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Final evaluation")
    print("=" * 50)
    evaluate(model, features, adj_norm, labels, idx_train, set_name="Train")
    evaluate(model, features, adj_norm, labels, idx_val,   set_name="Validation")
    evaluate(model, features, adj_norm, labels, idx_test,  set_name="Test")


if __name__ == "__main__":
    main()

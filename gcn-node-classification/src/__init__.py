# GCN Node Classification - Source Package
#
# Modules:
#   dataset.py      -- Load and split the Cora dataset
#   preprocess.py   -- Symmetric normalisation of the adjacency matrix
#   layers.py       -- GraphConvolution layer (from scratch)
#   model.py        -- GCN and DeepGCN architectures
#   train.py        -- Training loop with optional embedding snapshots
#   evaluate.py     -- Evaluation helpers
#   visualize.py    -- t-SNE, training curves, confusion matrix, graph animation
#   experiments.py  -- Hyperparameter sweeps and over-smoothing study
#   utils.py        -- Seed, accuracy, logging

# GCN for Semi-Supervised Node Classification on the Cora Dataset

A from-scratch PyTorch implementation of the Graph Convolutional Network (GCN) described in:

> Kipf, T.N. and Welling, M., 2017.  
> *Semi-Supervised Classification with Graph Convolutional Networks.*  
> International Conference on Learning Representations (ICLR).

---

## Problem — Semi-Supervised Node Classification

Given a citation graph where:
- nodes are scientific papers described by bag-of-words feature vectors,
- edges are citation links,
- only a small subset of nodes have labels,

the goal is to predict the category of every unlabelled node by jointly leveraging the graph structure and the node features.

## The GCN Idea

A Graph Convolutional Network generalises CNNs to graphs.  Each layer aggregates information from a node's neighbours using the rule:

$$
H^{(l+1)} = \sigma\!\bigl(\tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}\;H^{(l)}\;W^{(l)}\bigr)
$$

where $\tilde{A} = A + I$ adds self-loops and $\tilde{D}$ is its degree matrix.  Stacking two such layers with ReLU and dropout yields a simple yet surprisingly powerful classifier.

---

## Repository Structure

```
├── README.md
├──cora/                   ← cora.content & cora.cites here
├──gcn-node-classification/
    ├── requirements.txt
    ├── figures/            
    ├── src/
    │   ├── __init__.py
    │   ├── dataset.py          ← load & split Cora
    │   ├── preprocess.py       ← normalise adjacency
    │   ├── layers.py           ← GraphConvolution layer
    │   ├── model.py            ← 2-layer GCN
    │   ├── train.py            ← training loop
    │   ├── evaluate.py         ← evaluation helpers
    │   └── utils.py            ← seed, accuracy, logging
    └── gcn.ipynb      ← all-in-one Jupyter notebook
```

---


### Dependencies

| Package    | Version |
|------------|---------|
| Python     | ≥ 3.9   |
| PyTorch    | ≥ 1.9   |
| NumPy      | ≥ 1.21  |
| SciPy      | ≥ 1.7   |
| Matplotlib | ≥ 3.4   |

---

## Usage


### Run the notebook

Open `gcn.ipynb` in Jupyter or VS Code and run all cells.

---



## References

- Kipf, T.N. and Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
- Cora dataset: McCallum et al. (2000). *Automating the Construction of Internet Portals with Machine Learning.*

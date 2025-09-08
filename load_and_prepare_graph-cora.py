import dgl
import torch
import numpy as np
from dgl.data import CoraGraphDataset

# STEP ONE: prepare edges for 2PS Partitioner
dataset = CoraGraphDataset()
g = dataset[0]  # DGLGraph

# extract edges
src, dst = g.edges()
edge_index = torch.stack([src, dst], dim=0).numpy()

# Transpose for: source  target
edges = edge_index.T  # shape: (num_edges, 2)

# remove duplicates and self-loops
edges = edges[edges[:, 0] != edges[:, 1]]
edges = np.unique(edges, axis=0)

# save file for edges
np.savetxt("/mnt/2ps/graph_output/cora/cora.edgelist", edges, fmt="%d")

# STEP TWO: prepare whole graph in DGL format
num_nodes = g.num_nodes()

# Add node features and labels
g.ndata['feat'] = g.ndata['feat']
g.ndata['label'] = g.ndata['label']

# Create train/val/test masks
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']

g.ndata['train_mask'] = train_mask
g.ndata['val_mask'] = val_mask
g.ndata['test_mask'] = test_mask

# save graph
dgl.save_graphs("/mnt/2ps/graph_output/cora/graph.dgl", [g])

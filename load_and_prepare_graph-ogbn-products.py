import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np


# STEP ONE: prepare edges for 2PS Partitioner
print("Load Graph...")
dataset = DglNodePropPredDataset(name="ogbn-products")
graph, labels = dataset[0]

# extract edges
print("Create edge_index...")
src, dst = graph.edges()
edge_index = torch.stack([src, dst], dim=0).numpy()

# Transpose for: source  target
edges = edge_index.T  # shape: (num_edges, 2)

# remove duplicates and self-loops
print("Remove duplicates and self-loops...")
edges = edges[edges[:, 0] != edges[:, 1]]
edges = np.unique(edges, axis=0)

# save file for edges
print("Save edgelist...")
np.savetxt("/mnt/2ps/graph_output/ogbn-products/ogbn-products.edgelist", edges, fmt="%d")

# STEP TWO: prepare whole graph in DGL format
num_nodes = graph.num_nodes()

# Add node features and labels
graph.ndata['label'] = labels.squeeze()

# Create train/val/test masks
print("Create train/val/test masks...")
split_idx = dataset.get_idx_split()
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[split_idx["train"]] = True
val_mask[split_idx["valid"]] = True
test_mask[split_idx["test"]] = True

graph.ndata["train_mask"] = train_mask
graph.ndata["val_mask"] = val_mask
graph.ndata["test_mask"] = test_mask

# save graph
print("Save Graph...")
dgl.save_graphs("/mnt/2ps/graph_output/ogbn-products/graph.dgl", [graph])

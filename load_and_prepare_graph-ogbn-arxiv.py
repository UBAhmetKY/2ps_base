import dgl
import torch
from ogb.nodeproppred import NodePropPredDataset
import numpy as np


# STEP ONE: prepare edges for 2PS Partitioner
dataset = NodePropPredDataset(name="ogbn-arxiv")
graph, labels = dataset[0]

# extract edges
edge_index = graph["edge_index"]  # shape: (2, num_edges)

# Transpose for: source  target
edges = edge_index.T  # shape: (num_edges, 2)

# remove duplicates and self-loops
edges = edges[edges[:, 0] != edges[:, 1]]
edges = np.unique(edges, axis=0)

# save file for edges
np.savetxt(
    "/mnt/2ps/graph_output/ogbn-arxiv/ogbn-arxiv.edgelist", edges, fmt="%d"
)

# STEP TWO: prepare whole graph in DGL format
num_nodes = graph["num_nodes"]

# Convert edge_index to PyTorch tensors
src = torch.from_numpy(edge_index[0]).long()
dst = torch.from_numpy(edge_index[1]).long()
g = dgl.graph((src, dst), num_nodes=num_nodes)

# Add node features and labels
g.ndata["feat"] = torch.from_numpy(graph["node_feat"])
g.ndata["label"] = torch.from_numpy(labels).squeeze()

# Create train/val/test masks
split_idx = dataset.get_idx_split()
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[split_idx["train"]] = True
val_mask[split_idx["valid"]] = True
test_mask[split_idx["test"]] = True

g.ndata["train_mask"] = train_mask
g.ndata["val_mask"] = val_mask
g.ndata["test_mask"] = test_mask

# save graph
dgl.save_graphs("/mnt/2ps/graph_output/ogbn-arxiv/graph.dgl", [g])

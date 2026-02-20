import os
import json
import argparse
import torch as th
import dgl
import glob
from dgl.data.utils import load_graphs, save_graphs, save_tensors
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

GLOBAL_GRAPH = None
GLOBAL_PARTITIONS = None


def init_worker(graph, partitions):
    global GLOBAL_GRAPH, GLOBAL_PARTITIONS
    GLOBAL_GRAPH = graph
    GLOBAL_PARTITIONS = partitions



def load_partition_node_sets(partition_dir):
    node_files = sorted(
        glob.glob(os.path.join(partition_dir, "part*/nodes.txt")),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).replace("part", ""))
    )
    partitions = []
    all_nodes = set()
    for file in node_files:
        with open(file, 'r') as f:
            part_nodes = sorted([int(line.strip()) for line in f if line.strip()])
            partitions.append(part_nodes)
            all_nodes.update(part_nodes)
    return partitions, node_files


def compute_node_ownership(partitions):
    owner = {}
    for part_id, nodes in enumerate(partitions):
        for nid in nodes:
            if nid not in owner:
                owner[nid] = part_id
    return owner


def build_partition(part_id, part_nodes, output_dir, abs_out_dir, owner):
    global GLOBAL_GRAPH, GLOBAL_PARTITIONS
    g_orig = GLOBAL_GRAPH
    partitions = GLOBAL_PARTITIONS

    print(f"[{part_id}]: Start Partition.", flush=True)

    tensor_node_feats = ['adj', 'feat', 'label', 'train_mask', 'test_mask', 'val_mask']

    num_parts = len(partitions)
    node_ids = th.tensor(part_nodes, dtype=th.int64)
    num_nodes = len(part_nodes)

    # Load node features
    feat       = g_orig.ndata['feat'][node_ids]
    label      = g_orig.ndata['label'][node_ids]
    train_mask = g_orig.ndata['train_mask'][node_ids]
    val_mask   = g_orig.ndata['val_mask'][node_ids]
    test_mask  = g_orig.ndata['test_mask'][node_ids]

    start_opt = time.time()

    # Pre-convert partitions to sets for O(1) membership tests
    partition_sets = [set(nodes) for nodes in partitions]

    # O(1) global-node-id → local-index mapping
    node_to_idx = {nid: i for i, nid in enumerate(part_nodes)}

    # Collect all edges incident to our partition nodes (both directions)
    src_out, dst_out = g_orig.out_edges(node_ids)
    src_in,  dst_in  = g_orig.in_edges(node_ids)

    # Build per-node neighbor sets (union of both directions)
    neighbors_map = [set() for _ in range(num_nodes)]

    for src_g, dst_g in zip(src_out.numpy(), dst_out.numpy()):
        if src_g in node_to_idx:
            neighbors_map[node_to_idx[src_g]].add(int(dst_g))

    for src_g, dst_g in zip(src_in.numpy(), dst_in.numpy()):
        if dst_g in node_to_idx:
            neighbors_map[node_to_idx[dst_g]].add(int(src_g))

    # Build adjacency-to-partition matrix: adj[i][p] = 1 if node i has a neighbour in partition p
    adj = th.zeros((num_nodes, num_parts), dtype=th.int32)
    for i in range(num_nodes):
        neighbors = neighbors_map[i]
        for other_pid, part_set in enumerate(partition_sets):
            if neighbors & part_set:
                adj[i][other_pid] = 1

    end_opt = time.time()
    print(f"[{part_id}]: Process Time: {end_opt - start_opt:.4f}s", flush=True)

    # Build subgraph (only once)
    subgraph = dgl.node_subgraph(g_orig, node_ids)
    #subgraph.ndata.pop(dgl.NID, None)
    subgraph.edata.clear()
    subgraph.ndata.clear()

    subgraph.ndata['val_mask']   = val_mask
    subgraph.ndata['test_mask']  = test_mask
    subgraph.ndata['train_mask'] = train_mask
    subgraph.ndata['label']      = label
    subgraph.ndata['feat']       = feat
    subgraph.ndata['adj']        = adj

    assert len(part_nodes) == subgraph.number_of_nodes(), (
        f"[{part_id}]: expected {len(part_nodes)} nodes, "
        f"subgraph has {subgraph.number_of_nodes()} nodes"
    )

    part_dir = os.path.join(output_dir, f"part{part_id}")
    os.makedirs(part_dir, exist_ok=True)

    save_graphs(os.path.join(part_dir, "graph.dgl"), [subgraph])

    ndata = {key: subgraph.ndata[key] for key in tensor_node_feats}
    save_tensors(os.path.join(part_dir, "node_feats.dgl"), ndata)
    save_tensors(os.path.join(part_dir, "edge_feats.dgl"), {})

    # Compute halo nodes: neighbours outside this partition
    partition_set = set(part_nodes)
    halo_nodes = sorted({
        n
        for nbrs in neighbors_map
        for n in nbrs
        if n not in partition_set
    })

    print(f"[{part_id}]: Partition finish.", flush=True)
    return {
        "meta": {
            f"part-{part_id}": {
                "node_feats": os.path.join(abs_out_dir, f"part{part_id}/node_feats.dgl"),
                "edge_feats": os.path.join(abs_out_dir, f"part{part_id}/edge_feats.dgl"),
                "part_graph": os.path.join(abs_out_dir, f"part{part_id}/graph.dgl"),
            }
        },
        "halo_nodes": {part_id: halo_nodes},
    }


def write_metadata(graph_name, g_orig, num_parts, node_map, edge_map,
                   all_halo_nodes, output_dir, part_metadata, json_name):
    def format_list(lst, per_line):
            """Return a string representing the list with per_line values per line."""
            lines = []
            for i in range(0, len(lst), per_line):
                chunk = lst[i:i+per_line]
                lines.append(json.dumps(chunk)[1:-1])  # remove [ ] added by dumps
            return "[\n    " + ",\n    ".join(lines) + "\n]"

    metadata = {
        "graph_name":  graph_name,
        "num_nodes":   g_orig.number_of_nodes(),
        "num_edges":   g_orig.number_of_edges(),
        "part_method": "vertex_cut",
        "partitioner": "Twophase_v2",
        "num_parts":   num_parts,
        "halo_nodes":  all_halo_nodes,   # dict: part_id (str) → [halo node ids]
        "node_map":    format_list(node_map, 20),
        "edge_map":    format_list(edge_map, 20),
    }
    metadata.update(part_metadata)
    out_path = os.path.join(output_dir, json_name)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)
    print(f"[✓] JSON written → {out_path}")


def build_edge_map(g_orig, partitions):
    """
    Assign each edge to exactly one partition.
    Priority: the partition that owns the source node; fall back to destination.
    Edges where neither endpoint belongs to any partition are left as -1.
    """
    # Build global node → partition map
    node_to_part = {}
    for pid, nodes in enumerate(partitions):
        for nid in nodes:
            node_to_part.setdefault(nid, pid)   # first assignment wins

    src, dst = g_orig.edges()
    src_np = src.numpy()
    dst_np = dst.numpy()

    edge_map = np.full(g_orig.number_of_edges(), -1, dtype=np.int32)
    for eid, (s, d) in enumerate(zip(src_np, dst_np)):
        if s in node_to_part:
            edge_map[eid] = node_to_part[s]
        elif d in node_to_part:
            edge_map[eid] = node_to_part[d]

    return edge_map.tolist()


def main(args):
    print("Load Partition.", flush=True)
    partitions, _ = load_partition_node_sets(args.partition_dir)

    print("Load Graph.", flush=True)
    g_orig = load_graphs(args.graph_path)[0][0]

    print("Compute Node Ownership.", flush=True)
    owner = compute_node_ownership(partitions)

    abs_out_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_out_dir, exist_ok=True)

    print("Start ProcessPool for building Partitions.", flush=True)
    part_metadata = {}
    all_halo_nodes = {}   # part_id (str) → list of halo node ids

    with ProcessPoolExecutor(initializer=init_worker, initargs=(g_orig,partitions)) as executor:
        futures = {
            executor.submit(
                build_partition,
                pid,
                part_nodes,
                args.output_dir,
                abs_out_dir,
                owner,
            ): pid
            for pid, part_nodes in enumerate(partitions)
        }

        for future in as_completed(futures):
            result = future.result()
            part_metadata.update(result["meta"])
            all_halo_nodes.update({
                str(k): v for k, v in result["halo_nodes"].items()
            })

    print("Building edge map.", flush=True)
    edge_map = build_edge_map(g_orig, partitions)

    node_map = th.full((g_orig.number_of_nodes(),), -1, dtype=th.int32)
    for nid, pid in owner.items():
        node_map[nid] = pid

    print("Write Metadata.", flush=True)
    write_metadata(
        args.graph_name,
        g_orig,
        len(partitions),
        node_map.tolist(),
        edge_map,
        all_halo_nodes,
        args.output_dir,
        part_metadata,
        args.json_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TwoPhase partitions into DGL-compatible format"
    )
    parser.add_argument("--graph_path",     type=str, required=True,
                        help="Path to the original graph.dgl file")
    parser.add_argument("--partition_dir",  type=str, required=True,
                        help="Directory containing part*/nodes.txt files")
    parser.add_argument("--output_dir",     type=str, required=True,
                        help="Output directory for part*/graph.dgl & (node|edge)_feats.dgl")
    parser.add_argument("--json_name",      type=str, default="graph.json",
                        help="Name of the output JSON file, e.g. 'cora.json'")
    parser.add_argument("--graph_name",     type=str, default="graph",
                        help="Graph name stored in the JSON, e.g. 'cora'")

    args = parser.parse_args()
    main(args)

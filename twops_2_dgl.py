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

def init_worker(graph):
    global GLOBAL_GRAPH
    GLOBAL_GRAPH = graph

def load_partition_node_sets(partition_dir):
    node_files = sorted(glob.glob(os.path.join(partition_dir, "part*/nodes.txt")))
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

def build_partition(part_id, part_nodes, partitions, output_dir, abs_out_dir, owner):
    global GLOBAL_GRAPH
    g_orig = GLOBAL_GRAPH

    print(f"[{part_id}]:Start Partition.", flush=True)

    graph_node_feats = ['val_mask', 'test_mask', 'train_mask', 'label', 'feat', 'inner_node', 'lf', 'adj']
    tensor_node_feats = ['adj', 'lf', 'inner_node', 'feat', 'label', 'train_mask', 'test_mask', 'val_mask']

    num_parts = len(partitions)
    node_ids = th.tensor(part_nodes, dtype=th.int64)
    subgraph = dgl.node_subgraph(g_orig, node_ids)
    orig_nids = node_ids
    num_nodes = len(part_nodes)

    # Daten laden
    feat = g_orig.ndata['feat'][orig_nids]
    label = g_orig.ndata['label'][orig_nids]
    train_mask = g_orig.ndata['train_mask'][orig_nids]
    val_mask = g_orig.ndata['val_mask'][orig_nids]
    test_mask = g_orig.ndata['test_mask'][orig_nids]

    lf = th.ones(num_nodes, dtype=th.int32)
    inner_node = th.tensor([1 if owner[nid] == part_id else 0 for nid in part_nodes], dtype=th.int32)

    start_opt = time.time()
    # Pre-convert partitions to sets once
    partition_sets = [set(nodes) for nodes in partitions]

    # Create a mapping from global node ID to local index for O(1) lookup
    node_to_idx = {part_nodes[i]: i for i, _ in enumerate(part_nodes)}

    # Get all edges involving our partition nodes
    src_edges, dst_edges = g_orig.out_edges(node_ids)
    in_src, in_dst = g_orig.in_edges(node_ids)

    # Build neighbor sets efficiently using vectorized operations
    neighbors_map = [set() for _ in range(num_nodes)]

    src_np = src_edges.numpy()
    dst_np = dst_edges.numpy()
    for src_global, dst_global in zip(src_np, dst_np):
        if src_global in node_to_idx:
            neighbors_map[node_to_idx[src_global]].add(dst_global)

    in_src_np = in_src.numpy()
    in_dst_np = in_dst.numpy()
    for src_global, dst_global in zip(in_src_np, in_dst_np):
        if dst_global in node_to_idx:
            neighbors_map[node_to_idx[dst_global]].add(src_global)

    # Build adjacency matrix
    adj = th.zeros((num_nodes, num_parts), dtype=th.int32)
    for i in range(num_nodes):
        neighbors = neighbors_map[i]
        for other_pid in range(num_parts):
            if neighbors & partition_sets[other_pid]:  # faster set intersection check
                adj[i][other_pid] = 1
    end_opt = time.time()
    print(f"[{part_id}]:Process Time: {end_opt - start_opt:.4f}s", flush=True)

    part_dir = os.path.join(output_dir, f"part{part_id}")
    os.makedirs(part_dir, exist_ok=True)

    subgraph = dgl.node_subgraph(g_orig, node_ids)
    subgraph.ndata.pop(dgl.NID, None)
    subgraph.edata.clear()

    # Features in richtiger Reihenfolge für graph.dgl
    #for key in graph_node_feats:
    #    subgraph.ndata[key] = eval(key)

    subgraph.ndata.clear()
    subgraph.ndata['val_mask']   = val_mask
    subgraph.ndata['test_mask']  = test_mask
    subgraph.ndata['train_mask'] = train_mask
    subgraph.ndata['label']      = label
    subgraph.ndata['feat']       = feat
    subgraph.ndata['inner_node'] = inner_node
    subgraph.ndata['lf']         = lf
    subgraph.ndata['adj']        = adj


    print(f"[{part_id}]: expected {len(part_nodes)} nodes, subgraph has {subgraph.number_of_nodes()} nodes", flush=True)
    assert len(part_nodes) == subgraph.number_of_nodes()
    save_graphs(os.path.join(part_dir, "graph.dgl"), [subgraph])

    # node_feat.dgl mit separater Reihenfolge
    ndata = {key: subgraph.ndata[key] for key in tensor_node_feats}
    save_tensors(os.path.join(part_dir, "node_feats.dgl"), ndata)
    save_tensors(os.path.join(part_dir, "edge_feats.dgl"), {})  # leer

    print(f"[{part_id}]: Partition finish.", flush=True)
    return {
        f"part-{part_id}": {
            "node_feats": os.path.join(abs_out_dir, f"part{part_id}/node_feats.dgl"),
            "edge_feats": os.path.join(abs_out_dir, f"part{part_id}/edge_feats.dgl"),
            "part_graph": os.path.join(abs_out_dir, f"part{part_id}/graph.dgl")
        }
    }


def write_metadata(graph_name, g_orig, node_counts, output_dir, part_metadata, json_name):
    metadata = {
        "graph_name": graph_name,
        "num_nodes": g_orig.number_of_nodes(),
        "num_edges": g_orig.number_of_edges(),
        "part_method": "TwoPhase",
        "num_parts": len(node_counts),
        "halo_hops": 0,
        "node_map": node_counts,
        "edge_map": 0
    }
    metadata.update(part_metadata)
    with open(os.path.join(output_dir, json_name), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)
    print(f"[✓] JSON-Datei geschrieben → {json_name}")

def main(args):
    print("Load Partition.", flush=True)
    partitions, _ = load_partition_node_sets(args.partition_dir)
    print("Load Graph.", flush=True)
    g_orig = load_graphs(args.graph_path)[0][0]

    print("Compute Node Ownership.", flush=True)
    owner = compute_node_ownership(partitions)
    node_counts = [sum(1 for nid in owner.values() if nid == p) for p in range(len(partitions))]

    part_metadata = {}
    abs_out_dir = os.path.abspath(args.output_dir)

    print("Start ProcessPool for build Partitions.", flush=True)
    with ProcessPoolExecutor(initializer=init_worker, initargs=(g_orig,)) as executor:
        futures = []

        for pid, part_nodes in enumerate(partitions):
            futures.append(
                executor.submit(
                    build_partition,
                    pid,
                    part_nodes,
                    partitions,
                    args.output_dir,
                    abs_out_dir,
                    owner
                )
            )

        for future in as_completed(futures):
            result = future.result()
            part_metadata.update(result)

    print("Write Metadata.", flush=True)
    write_metadata(args.graph_name, g_orig, node_counts, args.output_dir, part_metadata, args.json_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Konvertiere TwoPhase-Partitionen in libra2dgl-kompatibles Format")
    parser.add_argument("--graph_path", type=str, required=True,
                        help="Pfad zur originalen graph.dgl Datei")
    parser.add_argument("--partition_dir", type=str, required=True,
                        help="Pfad zu den nodes.txt-Dateien (eine pro Partition)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Ausgabeverzeichnis für part*/graph.dgl & (node|edge)_feats.dgl")
    parser.add_argument("--json_name", type=str, default="cora.json",
                        help="Name der JSON-Ausgabedatei")
    parser.add_argument("--graph_name", type=str, default="cora",
                        help="Name des Graphs in der JSON-Datei")

    args = parser.parse_args()
    main(args)

import os
import json
import argparse
import torch as th
import dgl
import glob
from dgl.data.utils import load_graphs, save_graphs, save_tensors

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

def build_partition(part_id, part_nodes, partitions, g_orig, output_dir, part_metadata, abs_out_dir, owner):
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

    adj = th.zeros((num_nodes, num_parts), dtype=th.int32)
    for i, nid in enumerate(part_nodes):
        neighbors = set(g_orig.successors(nid).tolist() + g_orig.predecessors(nid).tolist())
        for other_pid, other_nodes in enumerate(partitions):
            if neighbors.intersection(other_nodes):
                adj[i][other_pid] = 1

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


    print(f"Partition {part_id}: expected {len(part_nodes)} nodes, subgraph has {subgraph.number_of_nodes()} nodes")
    assert len(part_nodes) == subgraph.number_of_nodes()
    save_graphs(os.path.join(part_dir, "graph.dgl"), [subgraph])

    # node_feat.dgl mit separater Reihenfolge
    ndata = {key: subgraph.ndata[key] for key in tensor_node_feats}
    save_tensors(os.path.join(part_dir, "node_feat.dgl"), ndata)
    save_tensors(os.path.join(part_dir, "edge_feat.dgl"), {})  # leer

    part_metadata[f"part-{part_id}"] = {
        "node_feats": os.path.join(abs_out_dir, f"part{part_id}/node_feat.dgl"),
        "edge_feats": os.path.join(abs_out_dir, f"part{part_id}/edge_feat.dgl"),
        "part_graph": os.path.join(abs_out_dir, f"part{part_id}/graph.dgl")
    }

    print(f"[✓] part{part_id} erfolgreich erzeugt.")

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

    print("Build Partitions.", flush=True)
    for pid, part_nodes in enumerate(partitions):
        print(f"Build Partition PID: {pid}", flush=True)
        build_partition(pid, part_nodes, partitions, g_orig, args.output_dir, part_metadata, abs_out_dir, owner)

    print("Write Metadata.", flush=True)
    write_metadata(args.graph_name, g_orig, node_counts, args.output_dir, part_metadata, args.json_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Konvertiere TwoPhase-Partitionen in libra2dgl-kompatibles Format")
    parser.add_argument("--graph_path", type=str, required=True,
                        help="Pfad zur originalen graph.dgl Datei")
    parser.add_argument("--partition_dir", type=str, required=True,
                        help="Pfad zu den nodes.txt-Dateien (eine pro Partition)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Ausgabeverzeichnis für part*/graph.dgl & node_feat.dgl")
    parser.add_argument("--json_name", type=str, default="cora.json",
                        help="Name der JSON-Ausgabedatei")
    parser.add_argument("--graph_name", type=str, default="cora",
                        help="Name des Graphs in der JSON-Datei")

    args = parser.parse_args()
    main(args)

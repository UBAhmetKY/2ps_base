import os
import json
import argparse
import torch as th
import dgl
from dgl.data.utils import load_graphs, save_graphs, save_tensors
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Globals shared across workers (set once per process via initializer) ──────
GLOBAL_GRAPH      = None
GLOBAL_PARTITIONS = None
GLOBAL_OWNER      = None          # ← added so it isn't pickled per-task


def init_worker(graph, partitions, owner):
    global GLOBAL_GRAPH, GLOBAL_PARTITIONS, GLOBAL_OWNER
    GLOBAL_GRAPH      = graph
    GLOBAL_PARTITIONS = partitions
    GLOBAL_OWNER      = owner     # shared reference, not re-pickled per submit()

def make_split_masks(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Creates random train/val/test masks when not provided by the graph."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)

    train_end = int(num_nodes * train_ratio)
    val_end   = int(num_nodes * (train_ratio + val_ratio))

    train_mask = th.zeros(num_nodes, dtype=th.bool)
    val_mask   = th.zeros(num_nodes, dtype=th.bool)
    test_mask  = th.zeros(num_nodes, dtype=th.bool)

    train_mask[perm[:train_end]]       = True
    val_mask[perm[train_end:val_end]]  = True
    test_mask[perm[val_end:]]          = True

    return train_mask, val_mask, test_mask

# ─────────────────────────────────────────────
#  Partition-file loaders
# ─────────────────────────────────────────────

def load_partition_edge_assignment(assignment_file):
    """
    Vertex-cut format – each line: <src> <dst> <part_id>
    """
    partition_edges: dict[int, list[tuple[int, int]]] = {}
    partition_nodes: dict[int, set[int]] = {}

    with open(assignment_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            src, dst, part_id = int(tokens[0]), int(tokens[1]), int(tokens[2])
            partition_edges.setdefault(part_id, []).append((src, dst))
            partition_nodes.setdefault(part_id, set()).add(src)
            partition_nodes.setdefault(part_id, set()).add(dst)

    num_parts = max(partition_nodes.keys()) + 1
    partitions = [sorted(partition_nodes.get(pid, set())) for pid in range(num_parts)]
    print(f"[loader] edge-assignment: {num_parts} partitions from '{assignment_file}'")
    return partitions, partition_edges


# ─────────────────────────────────────────────
#  Ownership
# ─────────────────────────────────────────────

def compute_node_ownership(partitions) -> dict[int, int]:
    """First partition that claims a node is its owner (relevant for vertex-cut)."""
    owner: dict[int, int] = {}
    for part_id, nodes in enumerate(partitions):
        for nid in nodes:
            if nid not in owner:
                owner[nid] = part_id
    return owner


# ─────────────────────────────────────────────
#  Core partition builder
# ─────────────────────────────────────────────

def build_partition(part_id, part_nodes, output_dir, abs_out_dir):
    """
    Builds one partition subgraph.  All large shared objects come from
    GLOBAL_* (set once per worker process by the initializer) so they
    are never re-serialised per task.
    """
    global GLOBAL_GRAPH, GLOBAL_PARTITIONS, GLOBAL_OWNER
    g_orig     = GLOBAL_GRAPH
    partitions = GLOBAL_PARTITIONS
    owner      = GLOBAL_OWNER

    print(f"[{part_id}] Start", flush=True)
    t0 = time.time()

    num_parts  = len(partitions)
    node_ids   = th.tensor(part_nodes, dtype=th.int64)
    num_nodes  = len(part_nodes)

    # ── Node features (single batched index) ─────────────────────────────
    if all(k in g_orig.ndata for k in ('train_mask', 'val_mask', 'test_mask')):
        train_mask = g_orig.ndata['train_mask'][node_ids]
        val_mask   = g_orig.ndata['val_mask'][node_ids]
        test_mask  = g_orig.ndata['test_mask'][node_ids]
    else:
        # Generate masks over the full graph once, then slice per partition
        full_train, full_val, full_test = make_split_masks(g_orig.number_of_nodes())
        train_mask = full_train[node_ids]
        val_mask   = full_val[node_ids]
        test_mask  = full_test[node_ids]
    feat       = g_orig.ndata['feat'][node_ids]
    label      = g_orig.ndata['label'][node_ids]

    # ── local-index lookup ────────────────────────────────────────────────
    node_to_idx: dict[int, int] = {nid: i for i, nid in enumerate(part_nodes)}
    partition_set = set(part_nodes)

    # ── Batched edge fetch (ONE call each) ────────────────────────────────
    src_out, dst_out = g_orig.out_edges(node_ids)   # directed out-edges
    src_in,  dst_in  = g_orig.in_edges(node_ids)    # directed in-edges

    src_out_np = src_out.numpy()
    dst_out_np = dst_out.numpy()
    src_in_np  = src_in.numpy()
    dst_in_np  = dst_in.numpy()

    # ── adj: which partitions does each local node have neighbours in? ────
    #    Use the already-fetched out-edges; no per-node queries.
    adj = th.zeros((num_nodes, num_parts), dtype=th.int32)
    for src_g, dst_g in zip(src_out_np, dst_out_np):
        local = node_to_idx.get(int(src_g))
        if local is not None:
            dst_part = owner.get(int(dst_g))
            if dst_part is not None:
                adj[local, dst_part] = 1

    # ── halo nodes: all out/in neighbours not in this partition ───────────
    #    Build from already-fetched edges (no extra queries).
    halo_set: set[int] = set()
    for src_g, dst_g in zip(src_out_np, dst_out_np):
        if int(src_g) in node_to_idx and int(dst_g) not in partition_set:
            halo_set.add(int(dst_g))
    for src_g, dst_g in zip(src_in_np, dst_in_np):
        if int(dst_g) in node_to_idx and int(src_g) not in partition_set:
            halo_set.add(int(src_g))
    halo_nodes = sorted(halo_set)

    print(f"[{part_id}] Edge processing: {time.time() - t0:.3f}s", flush=True)

    # ── Subgraph ──────────────────────────────────────────────────────────
    subgraph = dgl.node_subgraph(g_orig, node_ids)
    subgraph.edata.clear()
    subgraph.ndata.clear()
    subgraph.ndata['feat']       = feat
    subgraph.ndata['label']      = label
    subgraph.ndata['train_mask'] = train_mask
    subgraph.ndata['val_mask']   = val_mask
    subgraph.ndata['test_mask']  = test_mask
    subgraph.ndata['adj']        = adj

    assert subgraph.number_of_nodes() == num_nodes, (
        f"[{part_id}] expected {num_nodes} nodes, got {subgraph.number_of_nodes()}"
    )

    # ── Save ──────────────────────────────────────────────────────────────
    part_dir = os.path.join(output_dir, f"part{part_id}")
    os.makedirs(part_dir, exist_ok=True)

    save_graphs(os.path.join(part_dir, "graph.dgl"), [subgraph])
    save_tensors(os.path.join(part_dir, "node_feats.dgl"),
                 {k: subgraph.ndata[k] for k in ('adj','feat','label',
                                                  'train_mask','test_mask','val_mask')})
    save_tensors(os.path.join(part_dir, "edge_feats.dgl"), {})

    print(f"[{part_id}] Done in {time.time() - t0:.3f}s", flush=True)
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


# ─────────────────────────────────────────────
#  Metadata helpers
# ─────────────────────────────────────────────

def write_metadata(graph_name, g_orig, num_parts, node_map, edge_map,
                   all_halo_nodes, output_dir, part_metadata, json_name):
    metadata = {
        "graph_name":  graph_name,
        "num_nodes":   g_orig.number_of_nodes(),
        "num_edges":   g_orig.number_of_edges(),
        "part_method": "vertex_cut",
        "partitioner": "dbh_twophase_v2",
        "num_parts":   num_parts,
        "halo_nodes":  all_halo_nodes,
        "node_map":    node_map,
        "edge_map":    edge_map,
    }
    metadata.update(part_metadata)
    out_path = os.path.join(output_dir, json_name)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)
    print(f"[✓] JSON written → {out_path}")


def build_edge_map(g_orig, owner: dict[int, int]) -> list[int]:
    """
    Assign each edge to a partition by following its source node's owner.
    Falls back to destination owner if source is unknown (vertex-cut boundary).
    """
    src_np, dst_np = g_orig.edges()
    src_np = src_np.numpy()
    dst_np = dst_np.numpy()

    edge_map = np.full(g_orig.number_of_edges(), -1, dtype=np.int32)
    for eid, (s, d) in enumerate(zip(src_np, dst_np)):
        pid = owner.get(int(s))
        if pid is None:
            pid = owner.get(int(d), -1)
        edge_map[eid] = pid

    return edge_map.tolist()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main(args):
    print("Load Partition.", flush=True)
    partitions, _ = load_partition_edge_assignment(args.partition_file)

    print("Load Graph.", flush=True)
    g_orig = load_graphs(args.graph_path)[0][0]

    print("Compute Node Ownership.", flush=True)
    owner = compute_node_ownership(partitions)

    abs_out_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_out_dir, exist_ok=True)

    print("Start ProcessPool.", flush=True)
    part_metadata  = {}
    all_halo_nodes = {}

    # Pass `owner` through the initializer so it is shared per-process,
    # not pickled individually for every submitted task.
    with ProcessPoolExecutor(
        initializer=init_worker,
        initargs=(g_orig, partitions, owner),
    ) as executor:
        futures = {
            executor.submit(
                build_partition,
                pid,
                part_nodes,
                args.output_dir,
                abs_out_dir,
            ): pid
            for pid, part_nodes in enumerate(partitions)
        }
        for future in as_completed(futures):
            result = future.result()
            part_metadata.update(result["meta"])
            all_halo_nodes.update({str(k): v for k, v in result["halo_nodes"].items()})

    print("Building edge map.", flush=True)
    edge_map = build_edge_map(g_orig, owner)

    node_map = th.full((g_orig.number_of_nodes(),), -1, dtype=th.int32)
    for nid, pid in owner.items():
        node_map[nid] = pid

    print("Write Metadata.", flush=True)
    write_metadata(
        args.graph_name, g_orig, len(partitions),
        node_map.tolist(), edge_map, all_halo_nodes,
        args.output_dir, part_metadata, args.json_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert partitioner output into DGL-compatible format."
    )
    parser.add_argument("--graph_path",        type=str, required=True)
    parser.add_argument("--output_dir",        type=str, required=True)
    parser.add_argument("--json_name",         type=str, default="graph.json")
    parser.add_argument("--graph_name",        type=str, default="graph")
    parser.add_argument("--partition_file",    type=str, default=None)

    args = parser.parse_args()

    main(args)

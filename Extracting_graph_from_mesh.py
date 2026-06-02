import os
import re
import sys
import math
import numpy as np
from collections import defaultdict

# -----------------------------
# Parsing helpers
# -----------------------------

def _strip_comment(line: str) -> str:
    # Abaqus comments start with **
    if line.strip().startswith("**"):
        return ""
    return line.strip()

def _parse_header_kv(header_line: str):
    # Example: "*Element, type=C3D8R, elset=EALL"
    # Returns dict of key->value (lowercased keys)
    parts = [p.strip() for p in header_line.split(",")]
    kv = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip().lower()] = v.strip()
        else:
            kv[p.strip().lower()] = True
    return kv

def read_inp_mesh(inp_path: str):
    """
    Reads nodes, elements, and elsets from an Abaqus/Gmsh .inp file.

    Returns:
      nodes: dict[int] -> np.array([x,y,z], float64)
      elements: list of (elem_label:int, etype:str, conn:list[int])
      elsets: dict[str] -> set[int] (element labels)
    """
    nodes = {}
    elements = []
    elsets = defaultdict(set)

    cur = None
    cur_elem_type = None
    cur_elset_name = None
    elset_generate = False

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = _strip_comment(raw)
            if not line:
                continue

            if line.startswith("*"):
                hdr = line.upper()
                cur = None
                cur_elem_type = None
                cur_elset_name = None
                elset_generate = False

                if hdr.startswith("*NODE"):
                    cur = "NODE"
                elif hdr.startswith("*ELEMENT"):
                    cur = "ELEMENT"
                    kv = _parse_header_kv(line)
                    cur_elem_type = (kv.get("type", "") or "").upper()
                    # element blocks may also specify an elset, but we primarily use explicit *Elset
                elif hdr.startswith("*ELSET"):
                    cur = "ELSET"
                    kv = _parse_header_kv(line)
                    cur_elset_name = kv.get("elset", None)
                    if cur_elset_name is None:
                        # Sometimes it's "ELSET=NAME" with weird spacing; try regex
                        m = re.search(r"ELSET\s*=\s*([^,]+)", line, re.IGNORECASE)
                        if m:
                            cur_elset_name = m.group(1).strip()
                    if cur_elset_name is None:
                        raise RuntimeError(f"Found *Elset without elset name: {line}")

                    elset_generate = ("generate" in [k.lower() for k in kv.keys()] or kv.get("generate", False) is True)
                else:
                    cur = None
                continue

            # Data line
            if cur == "NODE":
                # id, x, y, z
                parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                if len(parts) < 4:
                    continue
                nid = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                nodes[nid] = np.array([x, y, z], dtype=np.float16)

            elif cur == "ELEMENT":
                parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                if len(parts) < 2:
                    continue
                eid = int(parts[0])
                conn = [int(p) for p in parts[1:]]
                elements.append((eid, cur_elem_type, conn))

            elif cur == "ELSET":
                # either list of eids, or generate lines: start, end, step
                name = cur_elset_name
                if elset_generate:
                    parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                    if len(parts) >= 2:
                        start = int(parts[0]); end = int(parts[1])
                        step = int(parts[2]) if len(parts) >= 3 else 1
                        for eid in range(start, end + 1, step):
                            elsets[name].add(int(eid))
                else:
                    parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                    for p in parts:
                        elsets[name].add(int(p))

    return nodes, elements, dict(elsets)


# -----------------------------
# Geometry helpers
# -----------------------------

def tet_volume(a, b, c, d) -> float:
    # |(b-a) dot ((c-a) x (d-a))| / 6
    return abs(np.dot((b - a), np.cross((c - a), (d - a)))) / 6.0

def element_centroid(conn, nodes):
    pts = np.array([nodes[n] for n in conn], dtype=np.float16)
    return pts.mean(axis=0)

def element_volume(etype: str, conn, nodes) -> float:
    """
    Supports:
      - C3D4 (tet4) exact
      - C3D8 / C3D8R (hex8) via tet decomposition (approx but usually good)
    Fallback: bbox volume
    """
    et = (etype or "").upper()

    pts = [nodes[n] for n in conn]
    pts = np.array(pts, dtype=np.float16)

    # C3D4 tetrahedron
    if "C3D4" in et and len(conn) >= 4:
        a, b, c, d = pts[0], pts[1], pts[2], pts[3]
        return float(tet_volume(a, b, c, d))

    # C3D8 hexahedron (approx decomposition into tets)
    # Typical Abaqus node order for C3D8:
    # 1-4 bottom, 5-8 top
    if "C3D8" in et and len(conn) >= 8:
        p = pts[:8]
        # One common 5-tet decomposition (works for convex-ish hexes)
        tets = [
            (0, 1, 3, 4),
            (1, 2, 3, 6),
            (1, 3, 4, 6),
            (1, 4, 5, 6),
            (3, 4, 6, 7),
        ]
        v = 0.0
        for (i, j, k, l) in tets:
            v += tet_volume(p[i], p[j], p[k], p[l])
        return float(v)

    # fallback: bbox volume (very rough)
    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)
    return float(max(0.0, (xmax - xmin) * (ymax - ymin) * (zmax - zmin)))


# -----------------------------
# Graph helpers
# -----------------------------

def build_node2elems(elements):
    node2elems = defaultdict(list)  # node_id -> [ei,...]
    for ei, (_, _, conn) in enumerate(elements):
        for n in conn:
            node2elems[n].append(ei)
    return node2elems

def build_edge_index(node2elems):
    edges = set()
    for n, L in node2elems.items():
        if len(L) < 2:
            continue
        # connect all pairs
        for a in range(len(L)):
            ia = L[a]
            for b in range(a + 1, len(L)):
                ib = L[b]
                if ia == ib:
                    continue
                if ia < ib:
                    edges.add((ia, ib))
                else:
                    edges.add((ib, ia))

    src = []
    dst = []
    for (i, j) in edges:
        src.append(i); dst.append(j)
        src.append(j); dst.append(i)
    return np.asarray([src, dst], dtype=np.int64)

def bbox_from_nodes(nodes_dict):
    pts = np.array(list(nodes_dict.values()), dtype=np.float16)
    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)
    return (float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax))

def boundary_nodes(nodes_dict, bbox, tol_factor=1e-6):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    Lx = max(1.0, abs(xmax - xmin))
    Ly = max(1.0, abs(ymax - ymin))
    Lz = max(1.0, abs(zmax - zmin))
    tol = tol_factor * max(Lx, Ly, Lz)

    b = set()
    for nid, p in nodes_dict.items():
        x, y, z = p
        if (abs(x - xmin) <= tol or abs(x - xmax) <= tol or
            abs(y - ymin) <= tol or abs(y - ymax) <= tol or
            abs(z - zmin) <= tol or abs(z - zmax) <= tol):
            b.add(int(nid))
    return b


# -----------------------------
# Phase grouping
# -----------------------------

def is_volume_named_set(name: str) -> bool:
    u = name.upper()
    return ("VOLUME" in u) or (re.search(r"\bVOL\b", u) is not None) or ("VOL_" in u) or ("_VOL" in u)

def base_phase_key(set_name: str) -> str:
    if "_" in set_name:
        return set_name.split("_", 1)[0]
    return set_name


def phase_onehot_from_key(phase_key: str):
    """
    One-hot phase encoding:
      [matrix, continuous_fiber, short_fiber, spherical_fiber]
    """
    k = phase_key.upper()

    if k == "MATRIX":
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16)

    if k.startswith("UD"):
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float16)

    if k.startswith("SF") or k.startswith("SFR"):
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float16)

    if k.startswith("PR"):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float16)

    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float16)

def phase_id_from_key(phase_key: str) -> float:
    k = phase_key.upper()
    if k == "MATRIX":
        return 1.0
    if k.startswith("UD"):
        return 2.0
    if k.startswith("PR"):
        return 3.0
    if k.startswith("SF") or k.startswith("SFR"):
        return 4.0
    return 0.0

def collect_phase_groups(elsets: dict, matrix_set_name="MATRIX", allow_prefixes=("UD", "SFR", "PR", "SF")):
    # find matrix set case-insensitively
    matrix_key = None
    for k in elsets.keys():
        if k.upper() == matrix_set_name.upper():
            matrix_key = k
            break
    if matrix_key is None:
        raise KeyError(f"Matrix element set '{matrix_set_name}' not found (case-insensitive)")

    phase2labels = {"MATRIX": set(int(e) for e in elsets[matrix_key])}

    for sname, labels in elsets.items():
        if sname == matrix_key:
            continue
        if is_volume_named_set(sname):
            continue

        su = sname.upper()
        keep = any(su.startswith(p.upper()) for p in allow_prefixes)
        if not keep:
            continue

        # detailed sets only: must contain underscore
        if "_" not in sname:
            continue

        base = base_phase_key(sname)
        phase2labels.setdefault(base, set()).update(int(e) for e in labels)

    phase_keys = sorted(phase2labels.keys())
    return phase2labels, phase_keys

def phase_key_for_element_label(eid: int, phase2labels: dict):
    # non-matrix first, sorted
    non_matrix = [k for k in phase2labels.keys() if k != "MATRIX"]
    non_matrix.sort()
    for k in non_matrix:
        if eid in phase2labels[k]:
            return k
    return "MATRIX"


def target_mask_from_phase_keys(elem_phase_key, target_phase_key: str):
    """
    Returns a node/element-wise target flag.

    Because this script represents graph nodes as finite elements, this mask has
    shape [num_elements]. A value of 1.0 means the element belongs to the phase
    for which you want to generate parameters; 0.0 means it is context.
    """
    return np.asarray(
        [1.0 if str(k).upper() == str(target_phase_key).upper() else 0.0 for k in elem_phase_key],
        dtype=np.float16,
    )


# -----------------------------
# Main export
# -----------------------------

def export_graphs_from_inp(stage: str, rve_id: str, mesh_id: str, out_dir: str = "."):


    inp_path = f'gmsh_stage_{stage}_rve_{rve_id}_mesh_{mesh_id}.inp'
    nodes, elements, elsets = read_inp_mesh(inp_path)

    bbox = bbox_from_nodes(nodes)
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    Lx = max(1e-30, xmax - xmin)
    Ly = max(1e-30, ymax - ymin)
    Lz = max(1e-30, zmax - zmin)

    bnodes = boundary_nodes(nodes, bbox)

    phase2labels, phase_keys = collect_phase_groups(elsets, matrix_set_name="MATRIX")
    print("Phase groups found:", phase_keys)

    Ne = len(elements)
    elem_labels = np.zeros((Ne,), dtype=np.int32)
    elem_phase_key = [""] * Ne

    # Build adjacency
    node2elems = build_node2elems(elements)
    edge_index_full = build_edge_index(node2elems)

    # x_full has 10 features:
    # [vol_n, cx_n, cy_n, cz_n, matrix, UD, SFR, PR, touches_boundary, is_target]
    #
    # The last column, is_target, is the new target flag for target-aware pooling.
    # In the generic full graph it is 0 for all elements. In the target-conditioned
    # full graphs saved below, it is 1 for elements belonging to the selected phase.
    feature_names = np.asarray([
        "vol_frac",
        "cx",
        "cy",
        "cz",
        "phase_matrix",
        "phase_ud",
        "phase_sfr",
        "phase_pr",
        "touches_boundary",
        "is_target",
    ])
    x_full = np.zeros((Ne, 10), dtype=np.float16)
    vol_raw = np.zeros((Ne,), dtype=np.float16)

    # First pass: raw volumes/centroids + phase ids + boundary flags
    for ei, (eid, etype, conn) in enumerate(elements):
        elem_labels[ei] = int(eid)

        pkey = phase_key_for_element_label(int(eid), phase2labels)
        elem_phase_key[ei] = pkey
        # pid = phase_id_from_key(pkey)
        phase_oh = phase_onehot_from_key(pkey)

        c = element_centroid(conn, nodes)
        vol = element_volume(etype, conn, nodes)

        vol_raw[ei] = vol

        # b0: touches boundary
        touches = 0.0
        for n in conn:
            if int(n) in bnodes:
                touches = 1.0
                break

        # normalize coords to [0,1]
        cx_n = (c[0] - xmin) / Lx
        cy_n = (c[1] - ymin) / Ly
        cz_n = (c[2] - zmin) / Lz

        x_full[ei, 0] = vol  # will normalize later if you want
        x_full[ei, 1] = cx_n
        x_full[ei, 2] = cy_n
        x_full[ei, 3] = cz_n
        x_full[ei, 4:8] = phase_oh
        x_full[ei, 8] = touches
        x_full[ei, 9] = 0.0  # is_target; filled in target-conditioned full graphs

    # Phase volumes from exclusive assignment (robust)
    V_tot = float(vol_raw.sum()) + 1e-30
    V_phase = {k: 0.0 for k in phase_keys}
    for ei in range(Ne):
        V_phase[elem_phase_key[ei]] += float(vol_raw[ei])

    # FVC in phase_keys order
    FVC_full = np.asarray([V_phase[k] / V_tot for k in phase_keys], dtype=np.float16)

    # Write node-wise fvc feature and normalize vol by V_tot (same as your latest file)
    fvc_map = {k: float(V_phase[k] / V_tot) for k in phase_keys}
    # x_full[:, 7] = np.array([fvc_map.get(elem_phase_key[ei], 0.0) for ei in range(Ne)], dtype=np.float16)
    x_full[:, 0] = x_full[:, 0] / V_tot  # normalized element volume fraction

    # Save full graph
    os.makedirs(out_dir, exist_ok=True)
    out_full = os.path.join(out_dir, f"graph_stage_{stage}_rve_{rve_id}_mesh_{mesh_id}.npz")
    np.savez_compressed(
        out_full,
        x=x_full,
        edge_index=edge_index_full,
        # elem_labels=elem_labels,
        # bbox=np.asarray(bbox, dtype=np.float16),
        FVC=FVC_full,
        phase_keys=np.asarray(phase_keys),
        feature_names=feature_names,
    )
    print("Saved full graph:", out_full)

    # Save one full graph per target phase.
    # These are the graphs you should use if you want the GNN to see all phases,
    # while the later pooling/readout focuses on one selected phase.
    for target_pkey in phase_keys:
        x_target = x_full.copy()
        target_mask = target_mask_from_phase_keys(elem_phase_key, target_pkey)
        x_target[:, 9] = target_mask

        safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target_pkey))
        out_target = os.path.join(
            out_dir,
            f"graph_stage_{stage}_rve_{rve_id}_mesh_{mesh_id}_target_{safe_target}.npz",
        )

        np.savez_compressed(
            out_target,
            x=x_target,
            edge_index=edge_index_full,
            FVC=FVC_full,
            target_mask=target_mask,
            target_phase=np.asarray([target_pkey]),
            phase_keys=np.asarray(phase_keys),
            feature_names=feature_names,
        )
        print(
            "Saved target-conditioned full graph:",
            out_target,
            "target_nodes=",
            int(target_mask.sum()),
            "total_nodes=",
            Ne,
        )

    # Map label -> index
    # label_to_ei = {int(elem_labels[i]): i for i in range(Ne)}
    #
    # # Save per-phase subgraphs
    # src_full = edge_index_full[0]
    # dst_full = edge_index_full[1]
    #
    # for pkey in phase_keys:
    #     labels = phase2labels[pkey]
    #     phase_eis = [label_to_ei[e] for e in labels if e in label_to_ei]
    #     phase_eis.sort()
    #     if len(phase_eis) == 0:
    #         print("Skipping empty phase:", pkey)
    #         continue
    #
    #     old2new = {old_i: new_i for new_i, old_i in enumerate(phase_eis)}
    #     phase_set = set(phase_eis)
    #
    #     # pid = phase_id_from_key(pkey)
    #     # fvc_val = float(V_phase.get(pkey, 0.0) / V_tot)
    #     #
    #     # # x_phase: keep same 8 columns (consistent with your newer output)
    #     # x_phase = np.zeros((len(phase_eis), 7), dtype=np.float16)
    #
    #     phase_oh = phase_onehot_from_key(pkey)
    #     fvc_val = float(V_phase.get(pkey, 0.0) / V_tot)
    #
    #     # x_phase: same feature size as x_full
    #     x_phase = np.zeros((len(phase_eis), 10), dtype=np.float16)
    #     elem_labels_phase = np.zeros((len(phase_eis),), dtype=np.int32)
    #
    #     for ii, old_i in enumerate(phase_eis):
    #         row = x_full[old_i].copy()
    #
    #         # ensure one-hot phase encoding is correct
    #         row[4:8] = phase_oh
    #
    #         # In a per-phase subgraph all included elements belong to the target phase.
    #         # This is mostly for backward compatibility; for context-aware pooling,
    #         # prefer the *_target_<phase>.npz full graphs saved above.
    #         row[9] = 1.0
    #
    #         x_phase[ii, :] = row
    #         elem_labels_phase[ii] = elem_labels[old_i]
    #
    #     # filter edges within phase
    #     src_p = []
    #     dst_p = []
    #     for k in range(src_full.shape[0]):
    #         s = int(src_full[k]); d = int(dst_full[k])
    #         if (s in phase_set) and (d in phase_set):
    #             src_p.append(old2new[s])
    #             dst_p.append(old2new[d])
    #     edge_index_p = np.asarray([src_p, dst_p], dtype=np.int64)
    #
    #     fvc_phase = np.asarray([fvc_val], dtype=np.float16)
    #     target_mask_phase = np.ones((len(phase_eis),), dtype=np.float16)
    #
    #     out_phase = os.path.join(out_dir, f"graph_stage_{stage}_rve_{rve_id}_mesh_{mesh_id}_phase_{pkey}.npz")
    #     np.savez_compressed(
    #         out_phase,
    #         x=x_phase,
    #         edge_index=edge_index_p,
    #         # elem_labels=elem_labels_phase,
    #         # bbox=np.asarray(bbox, dtype=np.float16),
    #         FVC=fvc_phase,
    #         target_mask=target_mask_phase,
    #         target_phase=np.asarray([pkey]),
    #         feature_names=feature_names,
    #         # phase_key=np.asarray([pkey])
    #     )
    #     print("Saved phase graph:", out_phase, "nodes=", x_phase.shape[0], "edges=", edge_index_p.shape[1])

    print("Done.")


if __name__ == "__main__":
    # Usage:
    #   python inp_to_graph_npz.py <inp_path> <stage> <rve_id> <mesh_id> [out_dir]
    # if len(sys.argv) < 5:
    #     raise SystemExit(
    #         "Usage:\n"
    #         "  python inp_to_graph_npz.py <inp_path> <stage> <rve_id> <mesh_id> [out_dir]\n"
    #     )
    # inp_path = sys.argv[1]
    stage = str(sys.argv[-3])
    rve_id = str(sys.argv[-2])
    mesh_id = str(sys.argv[-1])
    out_dir = "."



    export_graphs_from_inp(stage, rve_id, mesh_id, out_dir)
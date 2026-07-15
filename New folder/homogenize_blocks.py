from odbAccess import openOdb
from abaqusConstants import INTEGRATION_POINT
import os, sys, csv
import numpy as np

# ---------------------------
# CLI
# ---------------------------
# if len(sys.argv) < 3:
#     print("Usage: abaqus python homogenize_abaqus.py -- <main_folder> <strain_mag> [nx] [ny] [nz]")
#     sys.exit(1)
#
# MAIN_FOLDER = sys.argv[-2]
# STRAIN_MAG  = float(sys.argv[-1])
#
# # Default block split
# NX, NY, NZ = 3, 3, 3
#
# # Optional:
# # abaqus python homogenize_abaqus.py -- <main_folder> <strain_mag> 3 3 3
# if len(sys.argv) >= 6:
#     NX = int(sys.argv[-4])
#     NY = int(sys.argv[-3])
#     NZ = int(sys.argv[-2])
#     STRAIN_MAG = float(sys.argv[-1])
#     MAIN_FOLDER = sys.argv[-5]

MAIN_FOLDER = sys.argv[-2]
STRAIN_MAG  = float(sys.argv[-1])

# MAIN_FOLDER = 'D:/MSF_IMN/Training_data_generation/Training_data0117/'
# STRAIN_MAG = 0.01
NX, NY, NZ = 5,5,5
# Voigt order: [11,22,33,12,13,23]
STEP_TO_COL = {
    'Step-1': 0,  # E11
    'Step-2': 1,  # E22
    'Step-3': 2,  # E33
    'Step-4': 3,  # E12
    'Step-5': 4,  # E13
    'Step-6': 5,  # E23
}

WANT_STRESS = ['S11', 'S22', 'S33', 'S12', 'S13', 'S23']


def find_single_odb_in_folder(folder_path):
    candidates = [fn for fn in os.listdir(folder_path) if fn.lower().endswith('.odb')]
    if not candidates:
        return None, None

    folder_name = os.path.basename(folder_path)
    num = folder_name.split('_')[-1]

    for fn in candidates:
        if 'i_' in fn.lower():
            return os.path.join(folder_path, fn), num

    return os.path.join(folder_path, candidates[0]), num


def last_frame_of_step(odb, step_name):
    step = odb.steps[step_name]
    return step.frames[-1]


def build_instance_node_map(inst):
    node_map = {}
    for n in inst.nodes:
        node_map[n.label] = np.array(n.coordinates, dtype=float)
    return node_map


def get_rve_bbox(odb):
    """
    Global bounding box over all instance nodes.
    """
    mins = np.array([1e30, 1e30, 1e30], dtype=float)
    maxs = np.array([-1e30, -1e30, -1e30], dtype=float)

    for inst_name, inst in odb.rootAssembly.instances.items():
        for n in inst.nodes:
            xyz = np.array(n.coordinates, dtype=float)
            mins = np.minimum(mins, xyz)
            maxs = np.maximum(maxs, xyz)

    return mins, maxs


def build_blocks(bmin, bmax, nx, ny, nz):
    """
    Returns list of dicts:
      {
        'id': linear block id,
        'ijk': (ix,iy,iz),
        'min': [x0,y0,z0],
        'max': [x1,y1,z1]
      }
    """
    xs = np.linspace(bmin[0], bmax[0], nx + 1)
    ys = np.linspace(bmin[1], bmax[1], ny + 1)
    zs = np.linspace(bmin[2], bmax[2], nz + 1)

    blocks = []
    bid = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                blocks.append({
                    'id': bid,
                    'ijk': (ix, iy, iz),
                    'min': np.array([xs[ix], ys[iy], zs[iz]], dtype=float),
                    'max': np.array([xs[ix+1], ys[iy+1], zs[iz+1]], dtype=float),
                })
                bid += 1
    return blocks


def aabb_overlap(a_min, a_max, b_min, b_max, tol=1e-12):
    """
    Axis-aligned bounding-box overlap test.
    Touching boundary counts as overlap.
    """
    for d in range(3):
        if a_max[d] < b_min[d] - tol or a_min[d] > b_max[d] + tol:
            return False
    return True


def build_element_block_membership(odb, blocks):
    """
    For each element, compute its nodal bounding box and assign it to every
    block it overlaps.

    Returns:
      elem_to_blocks[(inst_name, elem_label)] = [block_ids...]
    """
    elem_to_blocks = {}

    for inst_name, inst in odb.rootAssembly.instances.items():
        node_map = build_instance_node_map(inst)

        for elem in inst.elements:
            conn = elem.connectivity
            coords = np.array([node_map[nlab] for nlab in conn], dtype=float)

            e_min = coords.min(axis=0)
            e_max = coords.max(axis=0)

            memberships = []
            for blk in blocks:
                if aabb_overlap(e_min, e_max, blk['min'], blk['max']):
                    memberships.append(blk['id'])

            elem_to_blocks[(inst_name, elem.label)] = memberships

    return elem_to_blocks


def avg_stress_IVOL_per_block(frame, elem_to_blocks, nblocks):
    fouts = frame.fieldOutputs
    if 'S' not in fouts:
        raise RuntimeError("No stress output 'S' found in ODB.")
    if 'IVOL' not in fouts:
        raise RuntimeError("No 'IVOL' found in ODB. Add IVOL to Field Output Requests.")

    S = fouts['S'].getSubset(position=INTEGRATION_POINT)
    V = fouts['IVOL'].getSubset(position=INTEGRATION_POINT)

    comp = S.componentLabels
    comp_index = {name: i for i, name in enumerate(comp)}

    vol_map = {}
    for v in V.values:
        key = (v.instance.name, v.elementLabel, getattr(v, 'integrationPoint', None))
        vol_map[key] = v.data

    # per block accumulators
    wsum = np.zeros((nblocks, 6), dtype=float)
    vtot = np.zeros(nblocks, dtype=float)

    for s in S.values:
        ip_key = (s.instance.name, s.elementLabel, getattr(s, 'integrationPoint', None))
        elem_key = (s.instance.name, s.elementLabel)

        vol = vol_map.get(ip_key, None)
        if vol is None:
            continue

        block_ids = elem_to_blocks.get(elem_key, [])
        if not block_ids:
            continue

        data = s.data
        sig6 = np.zeros(6, dtype=float)
        for k, cname in enumerate(WANT_STRESS):
            idx = comp_index.get(cname, None)
            if idx is not None:
                sig6[k] = data[idx]

        # If element overlaps multiple blocks, add it to all of them
        for bid in block_ids:
            wsum[bid, :] += sig6 * vol
            vtot[bid] += vol

    sigma_avg_blocks = np.full((nblocks, 6), np.nan, dtype=float)
    for bid in range(nblocks):
        if vtot[bid] > 0.0:
            sigma_avg_blocks[bid, :] = wsum[bid, :] / vtot[bid]

    return sigma_avg_blocks


def build_C_from_block_step_sigmas(block_step_sigma, nblocks):
    """
    block_step_sigma[step_name] = array(nblocks,6)

    Returns:
      C_blocks shape = (nblocks, 6, 6)
    """
    C_blocks = np.zeros((nblocks, 6, 6), dtype=float)

    for step_name, sigma_blocks in block_step_sigma.items():
        if step_name not in STEP_TO_COL:
            continue
        j = STEP_TO_COL[step_name]
        for bid in range(nblocks):
            if np.any(np.isnan(sigma_blocks[bid])):
                C_blocks[bid, :, j] = np.nan
            else:
                C_blocks[bid, :, j] = sigma_blocks[bid, :] / STRAIN_MAG

    return C_blocks


def write_block_csv(path, C_blocks, blocks):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['block_id', 'ix', 'iy', 'iz', 'row', 'E11', 'E22', 'E33', 'E12', 'E13', 'E23']
        w.writerow(header)
        rows = ['S11', 'S22', 'S33', 'S12', 'S13', 'S23']

        for blk in blocks:
            bid = blk['id']
            ix, iy, iz = blk['ijk']
            for i in range(6):
                w.writerow([bid, ix, iy, iz, rows[i]] + list(C_blocks[bid, i, :]))


# ---------------------------
# MAIN LOOP
# ---------------------------
folders = []
for name in os.listdir(MAIN_FOLDER):
    p = os.path.join(MAIN_FOLDER, name)
    if os.path.isdir(p) and name.lower().startswith('folder_'):
        folders.append(p)
folders.sort()

print("Main folder:", MAIN_FOLDER)
print("Found %d folder_* directories." % len(folders))
print("Block split = %d x %d x %d" % (NX, NY, NZ))

homo_blocks = {}
block_meta_written = False

for fdir in folders:
    folder_name = os.path.basename(fdir)
    number_str = folder_name.split('_')[-1]

    odb_path, _ = find_single_odb_in_folder(fdir)
    if odb_path is None:
        print("WARNING: No ODB found in", fdir)
        continue

    print("\n=== %s ===" % fdir)
    print("ODB:", os.path.basename(odb_path))

    odb = openOdb(odb_path, readOnly=True)
    try:
        # Build blocks once per ODB
        rve_min, rve_max = get_rve_bbox(odb)
        blocks = build_blocks(rve_min, rve_max, NX, NY, NZ)
        nblocks = len(blocks)

        print("RVE bbox min =", rve_min)
        print("RVE bbox max =", rve_max)
        print("Total blocks =", nblocks)

        elem_to_blocks = build_element_block_membership(odb, blocks)

        block_step_sigma = {}

        for sname in ['Step-1', 'Step-2', 'Step-3', 'Step-4', 'Step-5', 'Step-6']:
            if sname not in odb.steps:
                print("  WARNING: missing step", sname)
                continue

            frame = last_frame_of_step(odb, sname)
            sigma_blocks = avg_stress_IVOL_per_block(frame, elem_to_blocks, nblocks)
            block_step_sigma[sname] = sigma_blocks

            print("  %s done." % sname)

        C_blocks = build_C_from_block_step_sigmas(block_step_sigma, nblocks)
        homo_blocks[number_str] = C_blocks

        out_csv = os.path.join(fdir, "C_homogenized_blocks.csv")
        write_block_csv(out_csv, C_blocks, blocks)
        print("  Wrote:", out_csv)

        # save block metadata once
        if not block_meta_written:
            block_ids = np.array([blk['id'] for blk in blocks], dtype=int)
            block_ijk = np.array([blk['ijk'] for blk in blocks], dtype=int)
            block_min = np.array([blk['min'] for blk in blocks], dtype=float)
            block_max = np.array([blk['max'] for blk in blocks], dtype=float)

            np.savez(
                os.path.join(MAIN_FOLDER, "block_metadata.npz"),
                block_ids=block_ids,
                block_ijk=block_ijk,
                block_min=block_min,
                block_max=block_max
            )
            block_meta_written = True

    finally:
        odb.close()

np.savez(os.path.join(MAIN_FOLDER, 'homogenize_blocks.npz'), **homo_blocks)
print("\nDone.")
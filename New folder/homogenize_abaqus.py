from odbAccess import openOdb
from abaqusConstants import INTEGRATION_POINT
import os, re, sys, csv
import numpy as np

# ---------------------------
# CLI
# ---------------------------
if len(sys.argv) < 3:
    print("Usage: abaqus python homogenize_abaqus.py -- <main_folder> <strain_mag>")
    sys.exit(1)

# log = open('logdsjh.txt', 'w')

MAIN_FOLDER = sys.argv[-2]
STRAIN_MAG  = float(sys.argv[-1])
# stage = int(sys.argv[-2])
# graph_id = int(sys.argv[-1])
# rve = int(sys.argv[-1])

# Voigt order: [11,22,33,12,13,23]
STEP_TO_COL = {
    'Step-1': 0,  # E11
    'Step-2': 1,  # E22
    'Step-3': 2,  # E33
    'Step-4': 3,  # E12
    'Step-5': 4,  # E13
    'Step-6': 5,  # E23
}

def zero_C6():
    return [[0.0]*6 for _ in range(6)]


def find_single_odb_in_folder(folder_path):
    # Prefer I_<n>_stage_<stage>.odb, else I_<n>.odb, else first .odb
    candidates = [fn for fn in os.listdir(folder_path) if fn.lower().endswith('.odb')]
    if not candidates:
        return None

    folder_name = os.path.basename(folder_path)
    num = folder_name.split('_')[-1]


    # preferred_stage = f"I_{num}_{stage}_{graph_id}.odb"
    # preferred_plain  = f"I_{num}.odb"
    #
    # 1) try stage-specific
    for fn in candidates:
        if f'I_' in fn.lower():
            return os.path.join(folder_path, fn), num
    #
    # # 2) try plain
    # for fn in candidates:
    #     if fn.lower() == preferred_plain.lower():
    #         return os.path.join(folder_path, fn)

    # 3) fallback: first .odb
    return os.path.join(folder_path, candidates[0]), num


def last_frame_of_step(odb, step_name):
    step = odb.steps[step_name]
    return step.frames[-1]

def avg_stress_IVOL(frame):
    fouts = frame.fieldOutputs
    if 'S' not in fouts:
        raise RuntimeError("No stress output 'S' found in ODB.")
    if 'IVOL' not in fouts:
        raise RuntimeError("No 'IVOL' found in ODB. Add IVOL to Field Output Requests.")

    S = fouts['S'].getSubset(position=INTEGRATION_POINT)
    V = fouts['IVOL'].getSubset(position=INTEGRATION_POINT)

    comp = S.componentLabels
    comp_index = {name: i for i, name in enumerate(comp)}
    want = ['S11','S22','S33','S12','S13','S23']

    vol_map = {}
    for v in V.values:
        key = (v.instance.name, v.elementLabel, getattr(v, 'integrationPoint', None))
        vol_map[key] = v.data

    wsum = [0.0]*6
    vtot = 0.0

    for s in S.values:
        key = (s.instance.name, s.elementLabel, getattr(s, 'integrationPoint', None))
        vol = vol_map.get(key, None)
        if vol is None:
            continue

        data = s.data
        for k, cname in enumerate(want):
            idx = comp_index.get(cname, None)
            if idx is not None:
                wsum[k] += data[idx] * vol
        vtot += vol

    if vtot <= 0.0:
        raise RuntimeError("Total IVOL is 0. Check IVOL request.")


    # log.write(str(vtot) + '\n')

    return tuple(x/vtot for x in wsum)

def build_C_from_step_sigmas(step_sigma):
    # step_sigma: dict {'Step-1':sigma_avg(6,), ...}
    C = [[0.0]*6 for _ in range(6)]
    for step_name, sigma in step_sigma.items():
        if step_name not in STEP_TO_COL:
            continue
        j = STEP_TO_COL[step_name]

        for i in range(6):

            C[i][j] = sigma[i] / STRAIN_MAG
    return C

def write_C_csv(path, C):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['', 'E11','E22','E33','E12','E13','E23'])
        rows = ['S11','S22','S33','S12','S13','S23']
        for i in range(6):
            w.writerow([rows[i]] + C[i])

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

homo = dict()

for fdir in folders:
    folder_name = os.path.basename(fdir)
    number_str = folder_name.split('_')[-1]

    found = find_single_odb_in_folder(fdir)
    if found is None:
        print("WARNING: No ODB found in %s. Saving zero 6x6 matrix." % fdir)
        homo[number_str] = zero_C6()
        continue

    odb_path, _ = found

    print("\n=== %s ===" % fdir)
    print("ODB:", os.path.basename(odb_path))

    step_sigma = {}
    odb = None

    try:
        odb = openOdb(odb_path, readOnly=True)

        # Corrupt/incomplete ODBs sometimes open but contain no usable analysis steps.
        if len(odb.steps) == 0:
            raise RuntimeError("ODB has no steps")

        for sname in ['Step-1','Step-2','Step-3','Step-4','Step-5','Step-6']:
            if sname not in odb.steps:
                print("  WARNING: missing step", sname)
                continue

            step = odb.steps[sname]
            if len(step.frames) == 0:
                print("  WARNING: step %s has no frames" % sname)
                continue

            frame = last_frame_of_step(odb, sname)
            sig = avg_stress_IVOL(frame)
            step_sigma[sname] = sig
            print("  %s sigma_avg = %s" % (sname, str(sig)))

        # Require all six load cases. If any are missing, treat this RVE as failed.
        if len(step_sigma) != 6:
            raise RuntimeError("Only %d/6 steps had usable data" % len(step_sigma))

        homo[number_str] = build_C_from_step_sigmas(step_sigma)

    except Exception as e:
        print("  WARNING: Failed to homogenize %s. Reason: %s" % (folder_name, str(e)))
        print("  Saving zero 6x6 matrix for this folder.")
        homo[number_str] = zero_C6()

    finally:
        if odb is not None:
            try:
                odb.close()
            except Exception:
                pass

    # out_csv = os.path.join(fdir, "C_homogenized.csv")
    # write_C_csv(out_csv, C)
    # print("  Wrote:", out_csv)

np.savez(os.path.join(MAIN_FOLDER, f'homogenize.npz'), **homo)
print("\nDone.")


# import numpy as np
# np.savez(f'homo.npz', **homo)
# print("\nDone.")

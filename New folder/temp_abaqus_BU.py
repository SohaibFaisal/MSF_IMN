from abaqus import mdb
from abaqusConstants import *
import os
import sys
import numpy as np

# -------------------------
# USER SETTINGS
# -------------------------

modelName = 'msf'
partName  = 'RVE'
instName  = 'RVE-1'

graph_main = int(sys.argv[-4])
stage      = int(sys.argv[-3])
r          = int(sys.argv[-2])
graph_id   = int(sys.argv[-1])

base_inp = os.path.join(
    os.getcwd(),
    'gmsh_stage_%d_rve_%d_mesh_%d.inp' % (stage, r, graph_id)
)

strain = 0.01

load_cases = [
    [strain, 0,      0,      0,      0,      0     ],
    [0,      strain, 0,      0,      0,      0     ],
    [0,      0,      strain, 0,      0,      0     ],
    [0,      0,      0,      strain, 0,      0     ],
    [0,      0,      0,      0,      strain, 0     ],
    [0,      0,      0,      0,      0,      strain],
]

# -------------------------
# HELPERS
# -------------------------

def bbox_from_nodes(nodes):
    xs = [n.coordinates[0] for n in nodes]
    ys = [n.coordinates[1] for n in nodes]
    zs = [n.coordinates[2] for n in nodes]
    with open('loggggg.txt', 'w') as log:
        # log.write(str(xs))
        # log.write(str(ys))
        # log.write(str(zs))
        log.write(str(min(xs)))
        log.write('\n')
        log.write(str(min(ys)))
        log.write('\n')
        log.write(str(min(zs)))
        log.write('\n')
        log.write(str(max(xs)))
        log.write('\n')
        log.write(str(max(ys)))
        log.write('\n')
        log.write(str(max(zs)))
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def dist2(p, q):
    return (
        (p[0] - q[0])**2 +
        (p[1] - q[1])**2 +
        (p[2] - q[2])**2
    )


def _fmt_coef(c):
    if abs(c - round(c)) < 1e-12:
        return '%.1f' % c
    return '%.16g' % c


def eq_terms(term_list, triplets_per_line=4):
    lines = ['*EQUATION', str(len(term_list))]
    triplets = []

    for nid, dof, coef in term_list:
        if isinstance(nid, str):
            triplets.append('%s, %d, %s' % (nid, dof, _fmt_coef(coef)))
        else:
            triplets.append('%d, %d, %s' % (int(nid), dof, _fmt_coef(coef)))

    for i in range(0, len(triplets), triplets_per_line):
        lines.append(', '.join(triplets[i:i + triplets_per_line]))

    return '\n'.join(lines)


def make_rp_labels():
    return {
        'RP_E11': 9000001,
        'RP_E22': 9000002,
        'RP_E33': 9000003,
        'RP_E12': 9000004,
        'RP_E13': 9000005,
        'RP_E23': 9000006,
    }


def rp_for(dof, axis, rp):
    if axis == 'x':
        return {1: rp['RP_E11'], 2: rp['RP_E12'], 3: rp['RP_E13']}[dof]
    if axis == 'y':
        return {1: rp['RP_E12'], 2: rp['RP_E22'], 3: rp['RP_E23']}[dof]
    if axis == 'z':
        return {1: rp['RP_E13'], 2: rp['RP_E23'], 3: rp['RP_E33']}[dof]
    raise ValueError('axis must be x, y, or z')


def make_manual_rp_block(xmax, ymax, zmax, Lref, rp):
    x0 = xmax + 0.2 * Lref
    y0 = ymax + 0.2 * Lref
    z0 = zmax + 0.2 * Lref
    dx = 0.05 * Lref

    order = ['RP_E11', 'RP_E22', 'RP_E33', 'RP_E12', 'RP_E13', 'RP_E23']

    lines = []
    lines.append('** ----------------------------------------------------------------')
    lines.append('** Manual RP nodes for macro strain DOFs')
    lines.append('** ----------------------------------------------------------------')
    lines.append('*Node')

    for i, name in enumerate(order):
        lines.append('%d, %.16g, %.16g, %.16g' % (
            rp[name],
            x0 + i * dx,
            y0,
            z0
        ))

    for name in order:
        lines.append('*Nset, nset=%s' % name)
        lines.append('%d,' % rp[name])

    lines.append('** ---- End Manual RP nodes ----')

    return '\n'.join(lines) + '\n'


# def build_pbc_equations_general(
#         instName,
#         inst,
#         Lx, Ly, Lz,
#         xmin, xmax, ymin, ymax, zmin, zmax,
#         tol,
#         keyTol,
#         rp
#     ):
#
#     out = []
#     out.append('** ----------------------------------------------------------------')
#     out.append('** Periodic Boundary Conditions auto-generated')
#     out.append('** ----------------------------------------------------------------')
#
#     def q(v):
#         return int(round(v / keyTol))
#
#     def instnode(label):
#         return '%s.%d' % (instName, int(label))
#
#     def on_max(val, vmax):
#         return abs(val - vmax) <= tol
#
#     nodeByKey = {}
#
#     for n in inst.nodes:
#         x, y, z = n.coordinates
#         key = (q(x), q(y), q(z))
#
#         if key in nodeByKey:
#             old = nodeByKey[key]
#             raise RuntimeError(
#                 'Duplicate quantized node key. '
#                 'Nodes %d and %d map to %s. Reduce keyTol.'
#                 % (old.label, n.label, str(key))
#             )
#
#         nodeByKey[key] = n
#
#     n_equations = 0
#
#     for nP in inst.nodes:
#         x, y, z = nP.coordinates
#
#         hitX = on_max(x, xmax)
#         hitY = on_max(y, ymax)
#         hitZ = on_max(z, zmax)
#
#         if not (hitX or hitY or hitZ):
#             continue
#
#         xm = x - Lx if hitX else x
#         ym = y - Ly if hitY else y
#         zm = z - Lz if hitZ else z
#
#         keyM = (q(xm), q(ym), q(zm))
#
#         if keyM not in nodeByKey:
#             raise RuntimeError(
#                 'No periodic partner found for node %d: '
#                 '(%.16g, %.16g, %.16g) -> (%.16g, %.16g, %.16g)'
#                 % (nP.label, x, y, z, xm, ym, zm)
#             )
#
#         nM = nodeByKey[keyM]
#
#         deltas = []
#         if hitX:
#             deltas.append(('x', Lx))
#         if hitY:
#             deltas.append(('y', Ly))
#         if hitZ:
#             deltas.append(('z', Lz))
#
#         for dof in (1, 2, 3):
#             terms = [
#                 (instnode(nP.label), dof,  1.0),
#                 (instnode(nM.label), dof, -1.0),
#             ]
#
#             for axis, dL in deltas:
#                 terms.append((rp_for(dof, axis, rp), 1, -dL))
#
#             out.append(eq_terms(terms))
#             n_equations += 1
#
#     out.append('** Number of generated PBC equations: %d' % n_equations)
#     out.append('** ---- End PBC Equations ----')
#
#     return '\n'.join(out) + '\n'

def build_pbc_equations_general(
        instName,
        inst,
        Lx, Ly, Lz,
        xmin, xmax, ymin, ymax, zmin, zmax,
        tol,
        keyTol,
        rp
    ):

    out = []
    out.append('** ----------------------------------------------------------------')
    out.append('** Periodic Boundary Conditions auto-generated')
    out.append('** ----------------------------------------------------------------')

    Lref = max(Lx, Ly, Lz)
    matchTol = max(1e-6 * Lref, 1e-8)

    def instnode(label):
        return '%s.%d' % (instName, int(label))

    def on_max(val, vmax):
        return abs(val - vmax) <= tol

    def dist2_xyz(a, b):
        return (
            (a[0] - b[0])**2 +
            (a[1] - b[1])**2 +
            (a[2] - b[2])**2
        )

    all_nodes = list(inst.nodes)

    def find_partner(target):
        best = None
        best_d2 = None

        for n in all_nodes:
            d2 = dist2_xyz(n.coordinates, target)
            if best is None or d2 < best_d2:
                best = n
                best_d2 = d2

        if best is None:
            return None

        if best_d2 > matchTol**2:
            return None

        return best

    n_equations = 0

    for nP in all_nodes:
        x, y, z = nP.coordinates

        hitX = on_max(x, xmax)
        hitY = on_max(y, ymax)
        hitZ = on_max(z, zmax)

        if not (hitX or hitY or hitZ):
            continue

        xm = x - Lx if hitX else x
        ym = y - Ly if hitY else y
        zm = z - Lz if hitZ else z

        nM = find_partner((xm, ym, zm))

        if nM is None:
            raise RuntimeError(
                'No periodic partner found for node %d: '
                '(%.16g, %.16g, %.16g) -> (%.16g, %.16g, %.16g). '
                'matchTol=%.16g'
                % (nP.label, x, y, z, xm, ym, zm, matchTol)
            )

        deltas = []
        if hitX:
            deltas.append(('x', Lx))
        if hitY:
            deltas.append(('y', Ly))
        if hitZ:
            deltas.append(('z', Lz))

        for dof in (1, 2, 3):
            terms = [
                (instnode(nP.label), dof,  1.0),
                (instnode(nM.label), dof, -1.0),
            ]

            for axis, dL in deltas:
                terms.append((rp_for(dof, axis, rp), 1, -dL))

            out.append(eq_terms(terms))
            n_equations += 1

    out.append('** Number of generated PBC equations: %d' % n_equations)
    out.append('** ---- End PBC Equations ----')

    return '\n'.join(out) + '\n'

def make_manual_bc_steps(rp, load_cases):
    lines = []

    # Fix unused RP DOFs in Initial
    lines.append('** ----------------------------------------------------------------')
    lines.append('** Manual RP boundary conditions')
    lines.append('** ----------------------------------------------------------------')
    lines.append('*Boundary')
    for name in ['RP_E11', 'RP_E22', 'RP_E33', 'RP_E12', 'RP_E13', 'RP_E23']:
        lbl = rp[name]
        lines.append('%d, 2, 2, 0.' % lbl)
        lines.append('%d, 3, 3, 0.' % lbl)

    for i, lc in enumerate(load_cases):
        E11, E22, E33, E12, E13, E23 = lc

        lines.append('*Step, name=Step-%d, nlgeom=NO' % (i + 1))
        lines.append('*Static')
        lines.append('1., 1., 1e-05, 1.')

        lines.append('*Boundary')
        lines.append('%d, 1, 1, %.16g' % (rp['RP_E11'], E11))
        lines.append('%d, 1, 1, %.16g' % (rp['RP_E22'], E22))
        lines.append('%d, 1, 1, %.16g' % (rp['RP_E33'], E33))
        lines.append('%d, 1, 1, %.16g' % (rp['RP_E12'], E12))
        lines.append('%d, 1, 1, %.16g' % (rp['RP_E13'], E13))
        lines.append('%d, 1, 1, %.16g' % (rp['RP_E23'], E23))

        lines.append('*Output, field')
        lines.append('*Node Output')
        lines.append('U')
        lines.append('*Element Output, directions=YES')
        lines.append('S, IVOL')
        lines.append('*Output, history, variable=PRESELECT')
        lines.append('*End Step')

    return '\n'.join(lines) + '\n'


def strip_steps(lines):
    out = []
    i = 0

    while i < len(lines):
        if lines[i].strip().upper().startswith('*STEP'):
            i += 1
            while i < len(lines):
                if lines[i].strip().upper().startswith('*END STEP'):
                    i += 1
                    break
                i += 1
            continue

        out.append(lines[i])
        i += 1

    return out


def strip_auto_rp_and_pbc_blocks(lines):
    markers = [
        'Manual RP nodes for macro strain DOFs',
        'Periodic Boundary Conditions auto-generated',
        'Manual RP boundary conditions',
    ]

    text = ''.join(lines)

    for marker in markers:
        while marker in text:
            start = text.find('** ----------------------------------------------------------------', max(0, text.find(marker) - 100))
            if start < 0:
                start = text.find(marker)

            next_marker = text.find('** ----------------------------------------------------------------', text.find(marker) + len(marker))
            end_assembly = text.find('*End Assembly', text.find(marker))
            next_step = text.find('*Step', text.find(marker))

            candidates = [x for x in [next_marker, end_assembly, next_step] if x != -1]
            end = min(candidates) if candidates else len(text)

            text = text[:start] + text[end:]

    return text.splitlines(True)


def insert_into_assembly(inp_path, rp_block, eq_text):
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    lines = strip_auto_rp_and_pbc_blocks(lines)

    end_assembly_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith('*END ASSEMBLY'):
            end_assembly_idx = i
            break

    if end_assembly_idx is None:
        raise RuntimeError('Could not find *End Assembly in %s' % inp_path)

    insertion = (rp_block + eq_text).splitlines(True)
    lines = lines[:end_assembly_idx] + insertion + lines[end_assembly_idx:]

    with open(inp_path, 'w') as f:
        f.writelines(lines)


# def replace_steps_with_manual_steps(inp_path, step_text):
#     with open(inp_path, 'r') as f:
#         lines = f.readlines()
#
#     lines = strip_steps(lines)
#
#     lines.append('\n')
#     lines.extend(step_text.splitlines(True))
#
#     with open(inp_path, 'w') as f:
#         f.writelines(lines)

def replace_steps_with_manual_steps(inp_path, step_text):
    if step_text is None or len(step_text.strip()) == 0:
        raise RuntimeError("step_text is empty. Refusing to write inp without steps.")

    with open(inp_path, 'r') as f:
        lines = f.readlines()

    lines = strip_steps(lines)

    with open(inp_path, 'w') as f:
        f.writelines(lines)
        f.write('\n')
        f.write(step_text)
        f.write('\n')


# -------------------------
# IMPORT MODEL
# -------------------------

mdb.ModelFromInputFile(name=modelName, inputFileName=base_inp)

mdl = mdb.models[modelName]
prt = mdl.parts[partName]

# -------------------------
# MATERIALS / SECTIONS
# -------------------------

if 'MATRIX' not in mdl.materials:
    mdl.Material(name='MATRIX')
    mdl.materials['MATRIX'].Elastic(
        type=ORTHOTROPIC,
        table=((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),)
    )

if 'Section-MATRIX' not in mdl.sections:
    mdl.HomogeneousSolidSection(
        name='Section-MATRIX',
        material='MATRIX',
        thickness=None
    )

for name in ['UD', 'SFR', 'PR', 'COH-UD']:
    for x in range(1, 4):
        mat_name = name + str(x)
        sec_name = 'Section-' + mat_name

        if mat_name not in mdl.materials:
            mdl.Material(name=mat_name)
            mdl.materials[mat_name].Elastic(
                type=ORTHOTROPIC,
                table=((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),)
            )

        if sec_name not in mdl.sections:
            mdl.HomogeneousSolidSection(
                name=sec_name,
                material=mat_name,
                thickness=None
            )

# Assign sections
for k in list(prt.sets.keys()):
    print(k)

    if 'VOLUME' in k:
        prt.deleteSets(setNames=(k,))
        continue

    region = prt.sets[k]
    base_name = k.split('_')[0]
    section_name = 'Section-' + base_name

    if section_name not in mdl.sections:
        print('WARNING: section %s not found for set %s. Skipping.' % (section_name, k))
        continue

    try:
        prt.SectionAssignment(
            region=region,
            sectionName=section_name,
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField='',
            thicknessAssignment=FROM_SECTION
        )

        prt.MaterialOrientation(
            region=region,
            orientationType=GLOBAL,
            axis=AXIS_1,
            additionalRotationType=ROTATION_NONE,
            localCsys=None,
            fieldName='',
            stackDirection=STACK_3
        )
    except Exception as e:
        print('WARNING: failed section/orientation for set %s: %s' % (k, str(e)))

# -------------------------
# ASSEMBLY
# -------------------------

a = mdl.rootAssembly

if instName not in a.instances.keys():
    a.Instance(name=instName, part=prt, dependent=ON)

inst = a.instances[instName]

# -------------------------
# GRAPH-ONLY MODE
# -------------------------

if graph_main == 1:

    jobName = 'Graph_stage_%d_rve_%d_mesh_%d' % (stage, r, graph_id)

    if jobName in mdb.jobs:
        del mdb.jobs[jobName]

    mdb.Job(
        name=jobName,
        model=modelName,
        type=ANALYSIS,
        memory=90,
        memoryUnits=PERCENTAGE,
        getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        echoPrint=OFF,
        modelPrint=OFF,
        contactPrint=OFF,
        historyPrint=OFF,
        resultsFormat=ODB,
        numCpus=1,
        numGPUs=0
    )

    mdb.jobs[jobName].writeInput(consistencyChecking=OFF)

else:

    # -------------------------
    # GEOMETRY / TOLERANCES
    # -------------------------

    xmin, xmax, ymin, ymax, zmin, zmax = bbox_from_nodes(inst.nodes)

    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin
    Lref = max(Lx, Ly, Lz)

    tol = max(1e-2 * Lref, 1e-4)
    keyTol = max(1e-2 * Lref, 1e-4)

    print('Bounding box:')
    print('x:', xmin, xmax, 'Lx:', Lx)
    print('y:', ymin, ymax, 'Ly:', Ly)
    print('z:', zmin, zmax, 'Lz:', Lz)
    print('tol:', tol)
    print('keyTol:', keyTol)

    # -------------------------
    # ANCHOR ONLY MINIMUM CORNER
    # -------------------------

    targetA = (xmin, ymin, zmin)
    nA = min(inst.nodes, key=lambda n: dist2(n.coordinates, targetA))

    if 'N_ANCHOR_A' in a.sets:
        del a.sets['N_ANCHOR_A']

    a.Set(
        name='N_ANCHOR_A',
        nodes=inst.nodes.sequenceFromLabels(labels=(nA.label,))
    )

    if 'BC_ANCHOR_A' in mdl.boundaryConditions:
        del mdl.boundaryConditions['BC_ANCHOR_A']

    mdl.DisplacementBC(
        name='BC_ANCHOR_A',
        createStepName='Initial',
        region=a.sets['N_ANCHOR_A'],
        u1=0.0,
        u2=0.0,
        u3=0.0
    )

    print('Anchor node:', nA.label, nA.coordinates)

    # -------------------------
    # WRITE BASE INPUT
    # -------------------------

    jobName = 'Job_stage_%d_rve_%d_mesh_%d' % (stage, r, graph_id)

    if jobName in mdb.jobs:
        del mdb.jobs[jobName]

    mdb.Job(
        name=jobName,
        model=modelName,
        type=ANALYSIS,
        memory=90,
        memoryUnits=PERCENTAGE,
        getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        echoPrint=OFF,
        modelPrint=OFF,
        contactPrint=OFF,
        historyPrint=OFF,
        resultsFormat=ODB,
        numCpus=1,
        numGPUs=0
    )

    mdb.jobs[jobName].writeInput(consistencyChecking=OFF)

    inp_path = os.path.join(os.getcwd(), jobName + '.inp')

    # -------------------------
    # MANUAL RP NODES + PBC EQUATIONS + STEPS
    # -------------------------

    rp = make_rp_labels()

    step_text = make_manual_bc_steps(rp, load_cases)

    # Always write steps immediately after writeInput
    replace_steps_with_manual_steps(inp_path, step_text)

    rp_block = make_manual_rp_block(
        xmax=xmax,
        ymax=ymax,
        zmax=zmax,
        Lref=Lref,
        rp=rp
    )

    eq_text = build_pbc_equations_general(
        instName=instName,
        inst=inst,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        tol=tol,
        keyTol=keyTol,
        rp=rp
    )

    insert_into_assembly(inp_path, rp_block, eq_text)

    # final hard check
    with open(inp_path, "r") as f:
        txt = f.read()

    if "*Step" not in txt:
        raise RuntimeError("Final input file has no *Step: %s" % inp_path)

    if "Periodic Boundary Conditions auto-generated" not in txt:
        raise RuntimeError("Final input file has no PBC block: %s" % inp_path)


    replace_steps_with_manual_steps(inp_path, step_text)


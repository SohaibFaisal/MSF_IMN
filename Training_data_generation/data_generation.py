import numpy as np
import random as rnd
from pathlib import Path
import os
import gmsh
from math import isclose
import math


from .material_sampling_new import RangeSpec, sample_many_orthotropic_materials, sample_many_interface_materials







def split_annulus_face_transfinite(faceTag, cy, cz, ri, ro, xconst,
                                   nThetaQuarter=12, nRad=3):
    """
    faceTag: surface tag (dim=2) of the annulus face on x = xconst
    cy, cz: center in the yz-plane
    ri, ro: inner/outer radii
    nThetaQuarter: divisions per 90deg arc
    nRad: divisions across cohesive thickness
    """
    angles = [0, math.pi/2, math.pi, 3*math.pi/2]

    # Points on inner/outer circles in the face plane (x fixed)
    p_in, p_out = [], []
    for a in angles:
        p_in.append(gmsh.model.occ.addPoint(xconst, cy + ri*math.cos(a), cz + ri*math.sin(a)))
        p_out.append(gmsh.model.occ.addPoint(xconst, cy + ro*math.cos(a), cz + ro*math.sin(a)))

    # Radial lines
    radials = [gmsh.model.occ.addLine(p_in[i], p_out[i]) for i in range(4)]
    gmsh.model.occ.synchronize()

    # Fragment the annulus face with the radial lines -> 4 patches
    outDimTags, _ = gmsh.model.occ.fragment([(2, faceTag)], [(1, l) for l in radials],
                                            removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    patches = [t for (d, t) in outDimTags if d == 2]

    # Set transfinite constraints on each patch
    for s in patches:
        bnd = gmsh.model.getBoundary([(2, s)], oriented=False)
        curves = [tag for (d, tag) in bnd if d == 1]

        # Classify curves by radius using midpoint
        for c in curves:
            x, y, z = gmsh.model.getValue(1, c, [0.5])
            r = math.hypot(y - cy, z - cz)
            if abs(r - ri) < 1e-4 or abs(r - ro) < 1e-4:
                gmsh.model.mesh.setTransfiniteCurve(c, nThetaQuarter)
            else:
                gmsh.model.mesh.setTransfiniteCurve(c, nRad)

        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)

    return patches


def surfaces_on_plane(dim=2, axis="x", value=0.0, L=(1,1,1), eps=None):
    """
    Return surface tags whose bounding box touches the plane axis=value
    (within eps).
    """
    if eps is None:
        eps = 1e-8 * max(L)  # scale-aware tolerance

    tags = []
    for (d, s) in gmsh.model.getEntities(dim):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(d, s)
        if axis == "x":
            if isclose(xmin, value, abs_tol=eps) or isclose(xmax, value, abs_tol=eps):
                tags.append(s)
        elif axis == "y":
            if isclose(ymin, value, abs_tol=eps) or isclose(ymax, value, abs_tol=eps):
                tags.append(s)
        elif axis == "z":
            if isclose(zmin, value, abs_tol=eps) or isclose(zmax, value, abs_tol=eps):
                tags.append(s)
        else:
            raise ValueError("axis must be x, y, or z")
    return tags

def bbox_signature(surf_tag, axis, eps):
    """
    Signature based on bbox ranges in the two directions orthogonal to 'axis'.
    This is usually more stable than CenterOfMass for boolean-fragmented faces.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surf_tag)

    # quantize to eps to make hashing robust
    def q(v):
        return round(v / eps)

    if axis == "x":
        return (q(ymin), q(ymax), q(zmin), q(zmax))
    if axis == "y":
        return (q(xmin), q(xmax), q(zmin), q(zmax))
    if axis == "z":
        return (q(xmin), q(xmax), q(ymin), q(ymax))

def set_periodic_pairs(axis, master_tags, slave_tags, shift, L, eps=None):
    """
    Match master/slave by bbox signature and set periodicity.
    shift is (dx,dy,dz) from master -> slave.
    """
    if eps is None:
        eps = 1e-8 * max(L)

    master_map = {}
    for m in master_tags:
        sig = bbox_signature(m, axis, eps)
        master_map.setdefault(sig, []).append(m)

    # affine transform matrix (4x4) flattened row-major for gmsh
    dx, dy, dz = shift
    T = [1,0,0,dx,
         0,1,0,dy,
         0,0,1,dz,
         0,0,0,1]

    used = set()
    for s in slave_tags:
        sig = bbox_signature(s, axis, eps)
        candidates = master_map.get(sig, [])
        if not candidates:
            # If you hit this often, your geometry is not perfectly periodic/imprinted.
            continue

        # Usually there should be exactly one candidate; take first unused
        m = next((c for c in candidates if c not in used), candidates[0])
        used.add(m)

        gmsh.model.mesh.setPeriodic(2, [s], [m], T)

def apply_periodic(Lx, Ly, Lz):
    L = (Lx, Ly, Lz)
    eps = 1e-8 * max(L)   # scale-aware tolerance

    gmsh.model.occ.synchronize()  # IMPORTANT: do once before querying model entities

    # Collect surfaces on each boundary plane
    x0 = surfaces_on_plane(axis="x", value=0.0,   L=L, eps=eps)
    x1 = surfaces_on_plane(axis="x", value=Lx,    L=L, eps=eps)
    y0 = surfaces_on_plane(axis="y", value=0.0,   L=L, eps=eps)
    y1 = surfaces_on_plane(axis="y", value=Ly,    L=L, eps=eps)
    z0 = surfaces_on_plane(axis="z", value=0.0,   L=L, eps=eps)
    z1 = surfaces_on_plane(axis="z", value=Lz,    L=L, eps=eps)

    # Define master->slave shifts (choose your convention)
    set_periodic_pairs("x", master_tags=x0, slave_tags=x1, shift=(Lx, 0, 0), L=L, eps=eps)
    set_periodic_pairs("y", master_tags=y0, slave_tags=y1, shift=(0, Ly, 0), L=L, eps=eps)
    set_periodic_pairs("z", master_tags=z0, slave_tags=z1, shift=(0, 0, Lz), L=L, eps=eps)

# Example:
# apply_periodic(rve_size[0], rve_size[1], rve_size[2])

def check_collision_gmsh( fnc_new_origin, fnc_new_direction_vector, fnc_new_length, fnc_new_diameter,fnc_fiber_shape,fiber_info,rve_info):
    #print('Checking collision gmsh ...')
    gmsh.model.setCurrent("collision_check")

    #for rm3 in gmsh.model.getEntities(3):
    #    gmsh.model.remove_entities([rm3],True)

    if (fnc_fiber_shape == 'cylinder') or (fnc_fiber_shape == 'UD'):
        new_fiber = gmsh.model.occ.addCylinder(fnc_new_origin[0]-fnc_new_length/2*fnc_new_direction_vector[0],
                                   fnc_new_origin[1]-fnc_new_length/2*fnc_new_direction_vector[1],
                                   fnc_new_origin[2]-fnc_new_length/2*fnc_new_direction_vector[2],
                                   fnc_new_length*fnc_new_direction_vector[0],
                                   fnc_new_length*fnc_new_direction_vector[1],
                                   fnc_new_length*fnc_new_direction_vector[2], fnc_new_diameter/2)
    elif fnc_fiber_shape == 'sphere':
        new_fiber = gmsh.model.occ.addSphere(fnc_new_origin[0], fnc_new_origin[1], fnc_new_origin[2], fnc_new_diameter/2)

    gmsh.model.occ.synchronize()
    for a in range(1,new_fiber):
        out, _ = gmsh.model.occ.cut([(3, new_fiber)], [(3, int(a))],-1,False,False)
        if out == []:
            gmsh.model.occ.remove([(3, new_fiber)])
            gmsh.model.occ.synchronize()
            return True
        elif out[0][1] != new_fiber:
            gmsh.model.occ.remove([(3, new_fiber)])
            gmsh.model.occ.remove(out)
            gmsh.model.occ.synchronize()
            return True
    return False


def check_collision_gmsh_new( fnc_new_origin, fnc_new_direction_vector, fnc_new_length, fnc_new_diameter,fnc_fiber_shape,all_fiber_info,fiber_info, rve_info):

    rve_size = rve_info['MATRIX']['size']
    # print('Checking collision')
    # for fi in chain([fiber_info], all_fiber_info):
    for fi in all_fiber_info+[fiber_info]:
        # print(fi)
        for k,v in fi.items():
            if v[0][0] == -1:
                return False
            else:
                if fnc_fiber_shape == 'UD':
                    if v[-1] == 'yes_periodicity':
                        for p in v[6]:
                            new_origin = fi['p' + str(k) + ',' + str(p[3]) + ',' + str(p[4]) + ',' + str(p[5])][0]
                            distance = ((new_origin[1] - fnc_new_origin[1]) ** 2 + (new_origin[2] - fnc_new_origin[2]) ** 2) ** 0.5
                            if distance < (fnc_new_diameter + v[1]) * 1.0001/2:
                                return True
                    distance = ((v[0][2]-fnc_new_origin[2])**2 + (v[0][1]-fnc_new_origin[1])**2)**0.5
                    if distance < (fnc_new_diameter + v[1]) * 1.0001/2:
                        return True

                elif fnc_fiber_shape == 'cylinder':
                    if v[-1] == 'yes_periodicity':
                        for p in v[6]:
                            new_origin = fi['p' + str(k) + ',' + str(p[3]) + ',' + str(p[4]) + ',' + str(p[5])][0]
                            new_length = v[2]
                            distance = ((new_origin[1] - fnc_new_origin[1]) ** 2 + (new_origin[2] - fnc_new_origin[2]) ** 2) ** 0.5
                            if distance < (fnc_new_diameter + v[1]) * 1.01/2:
                                new_dist = abs(new_origin[0] - fnc_new_origin[0])  # FIXEDDD
                                if new_dist < (fnc_new_length+new_length) * 1.01/2:

                                    return True


                    distance = ((v[0][2]-fnc_new_origin[2])**2 + (v[0][1]-fnc_new_origin[1])**2)**0.5
                    if distance < (fnc_new_diameter + v[1]) * 1.01/2:
                        new_dist = abs(v[0][0]- fnc_new_origin[0])
                        if new_dist < (fnc_new_length+ v[2]) * 1.01/2:

                            return True


                elif fnc_fiber_shape == 'sphere':

                    if v[-1] == 'yes_periodicity':
                        for p in v[6]:
                            new_origin = fi['p' + str(k) + ',' + str(p[3]) + ',' + str(p[4]) + ',' + str(p[5])][0]
                            if 'PR' in v[-3]:
                                distance = ((new_origin[0] - fnc_new_origin[0]) ** 2 + (new_origin[1] - fnc_new_origin[1]) ** 2 + (new_origin[2] - fnc_new_origin[2]) ** 2) ** 0.5
                            else:
                                distance = ((new_origin[1] - fnc_new_origin[1]) ** 2 + (new_origin[2] - fnc_new_origin[2]) ** 2) ** 0.5


                            if distance < (fnc_new_diameter + v[1]) * 1.0001/2:


                                return True


                    if 'PR' in v[-3]:
                        distance = ((v[0][0]-fnc_new_origin[0])**2 + (v[0][2]-fnc_new_origin[2])**2 + (v[0][1]-fnc_new_origin[1])**2)**0.5
                    else:
                        distance = ((v[0][2] - fnc_new_origin[2]) ** 2 + (v[0][1] - fnc_new_origin[1]) ** 2) ** 0.5
                    if distance < (fnc_new_diameter + v[1]) * 1.0001/2:

                        return True






def clean_collision_check(a):
    if a == 0:
        return
    gmsh.model.setCurrent("collision_check")
    for xx in gmsh.model.occ.getEntities(3)[-a:]:
        gmsh.model.occ.remove([xx])
    gmsh.model.occ.synchronize()




def check_periodicity_gmsh(fnc_new_origin, fnc_new_direction_vector,fnc_new_length, fnc_new_diameter, fnc_rve, fnc_fiber_shape):
    periodic_origins = []
    final_periodic_origins = []
    periodic_possibilities = [-1, 0, 1]
    min_distance = 0.01
    #print('checking periodicity gmsh ...')
    gmsh.model.setCurrent("periodicity_check")
    # for rm3 in gmsh.model.getEntities(3):
    #     gmsh.model.remove_entities([rm3],True)
    gmsh.model.removeEntities(gmsh.model.getEntities(), recursive=True)
    gmsh.model.occ.synchronize()

    if (fnc_fiber_shape == 'cylinder') :
        fiber = gmsh.model.occ.addCylinder(fnc_new_origin[0]-fnc_new_length/2*fnc_new_direction_vector[0],
                                   fnc_new_origin[1]-fnc_new_length/2*fnc_new_direction_vector[1],
                                   fnc_new_origin[2]-fnc_new_length/2*fnc_new_direction_vector[2],
                                   fnc_new_length*fnc_new_direction_vector[0],
                                   fnc_new_length*fnc_new_direction_vector[1],
                                   fnc_new_length*fnc_new_direction_vector[2], fnc_new_diameter/2)

    elif fnc_fiber_shape == 'UD':
        direction = [1, 0, 0]
        fiber = gmsh.model.occ.addCylinder(fnc_new_origin[0]-fnc_rve[0]/2,
                                   fnc_new_origin[1],
                                   fnc_new_origin[2],
                                   fnc_rve[0]*direction[0],
                                   fnc_rve[1]*direction[1],
                                   fnc_rve[2]*direction[2], fnc_new_diameter/2)

    elif fnc_fiber_shape == 'sphere':
        fiber = gmsh.model.occ.addSphere(fnc_new_origin[0], fnc_new_origin[1], fnc_new_origin[2], fnc_new_diameter / 2)

    if fnc_fiber_shape == 'UD':

        box = gmsh.model.occ.addBox(-fnc_rve[0], 0, 0, fnc_rve[0]*3, fnc_rve[1], fnc_rve[2])
    else:
        box = gmsh.model.occ.addBox(0, 0, 0, fnc_rve[0], fnc_rve[1], fnc_rve[2])
    gmsh.model.occ.synchronize()
    vol = gmsh.model.occ.getMass(3, fiber)
    a = gmsh.model.occ.cut([(3, fiber)], [(3, box)], -1, True, True)

    gmsh.model.occ.synchronize()
    if a[0] == []:
        gmsh.model.occ.synchronize()
        return []
    else:
        new_vol = gmsh.model.occ.getMass(3, a[0][0][1])
        if new_vol > 0.96*vol or new_vol < 0.04*vol:
            gmsh.model.occ.synchronize()
            return False

        gmsh.model.occ.remove([(3, box)])
        box = gmsh.model.occ.addBox(0, 0, 0, fnc_rve[0], fnc_rve[1], fnc_rve[2])
        if fnc_fiber_shape == 'UD':


            for y in periodic_possibilities:
                for z in periodic_possibilities:
                    if y == 0 and z == 0:
                        continue
                    else:
                        xx = fnc_new_origin[0]
                        yy = fnc_new_origin[1] + y * fnc_rve[1]
                        zz = fnc_new_origin[2] + z * fnc_rve[2]
                        periodic_origins.append([xx, yy, zz, 0,y,z])

        else:

            for x in periodic_possibilities:
                for y in periodic_possibilities:
                    for z in periodic_possibilities:
                        if x == 0 and y == 0 and z == 0:
                            continue
                        else:
                            xx = fnc_new_origin[0] + x * fnc_rve[0]
                            yy = fnc_new_origin[1] + y * fnc_rve[1]
                            zz = fnc_new_origin[2] + z * fnc_rve[2]
                            periodic_origins.append([xx, yy, zz, x, y, z])

        if (fnc_fiber_shape == 'cylinder') or (fnc_fiber_shape == 'UD'):
            for p in periodic_origins:
                p_cyl = gmsh.model.occ.addCylinder(p[0]-fnc_new_length/2*fnc_new_direction_vector[0],
                                   p[1]-fnc_new_length/2*fnc_new_direction_vector[1],
                                   p[2]-fnc_new_length/2*fnc_new_direction_vector[2],
                                   fnc_new_length*fnc_new_direction_vector[0],
                                   fnc_new_length*fnc_new_direction_vector[1],
                                   fnc_new_length*fnc_new_direction_vector[2], fnc_new_diameter/2)
                b = gmsh.model.occ.intersect([(3, p_cyl)],[(3,box)],-1,True,False)
                if b[0] == []:
                    gmsh.model.occ.remove([(3, p_cyl)],True)
                else:
                    final_periodic_origins.append(p)
                    gmsh.model.occ.remove([(3, p_cyl)], True)
        elif fnc_fiber_shape == 'sphere':
            for p in periodic_origins:
                p_sph = gmsh.model.occ.addSphere(p[0], p[1], p[2], fnc_new_diameter / 2)
                b = gmsh.model.occ.intersect([(3, p_sph)], [(3, box)], -1, True, False)
                if b[0] == []:
                    gmsh.model.occ.remove([(3, p_sph)], True)
                else:
                    final_periodic_origins.append(p)
                    gmsh.model.occ.remove([(3, p_sph)], True)

        gmsh.model.occ.remove([(3,box)], True)
        gmsh.model.occ.synchronize()
        return final_periodic_origins



def check_periodicity_new(fnc_new_origin, fnc_new_direction_vector,fnc_new_length, fnc_new_diameter, fnc_rve, fnc_fiber_shape):
    periodic_origins = []
    final_periodic_origins = []
    periodic_possibilities = [-1, 0, 1]
    min_distance = 0.01

    # []: No periodicity
    # False: Wrong periodicity
    # True: Yes periodiciity
    periodic = True
    if fnc_fiber_shape == 'UD':
        if fnc_new_origin[1] > fnc_rve[1]-fnc_new_diameter/2.0001 or fnc_new_origin[1] < fnc_new_diameter/2.0001 or fnc_new_origin[2] > fnc_rve[2]-fnc_new_diameter/2.0001 or fnc_new_origin[2] < fnc_new_diameter/2.0001:
            for y in periodic_possibilities:
                for z in periodic_possibilities:
                    if y == 0 and z == 0:
                        continue
                    else:
                        xx = fnc_new_origin[0]
                        yy = fnc_new_origin[1] + y * fnc_rve[1]
                        zz = fnc_new_origin[2] + z * fnc_rve[2]
                        periodic_origins.append([xx, yy, zz, 0,y,z])
            for p in periodic_origins:
                new_origin = [p[0], p[1], p[2]]
                if new_origin[1] > -fnc_new_diameter / 2.0001 and new_origin[1] < fnc_rve[1] + fnc_new_diameter / 2.0001 and new_origin[2] < fnc_rve[
                    2] + fnc_new_diameter / 2.0001 and new_origin[2] > -fnc_new_diameter / 2.0001:
                    final_periodic_origins.append(p)

            return final_periodic_origins
        else:
            return []

    elif fnc_fiber_shape == 'cylinder':
        if fnc_new_origin[1] > fnc_rve[1]-fnc_new_diameter/2.0001 or fnc_new_origin[1] < fnc_new_diameter/2.0001 or fnc_new_origin[2] > fnc_rve[2]-fnc_new_diameter/2.0001 or fnc_new_origin[2] < fnc_new_diameter/2.0001 or fnc_new_origin[0]<fnc_new_length/2.0001 or fnc_new_origin[0]>fnc_rve[0]- fnc_new_length/2.0001:
            for x in periodic_possibilities:
                for y in periodic_possibilities:
                    for z in periodic_possibilities:
                        if x == 0 and y == 0 and z == 0:
                            continue
                        else:
                            xx = fnc_new_origin[0] + x * fnc_rve[0]
                            yy = fnc_new_origin[1] + y * fnc_rve[1]
                            zz = fnc_new_origin[2] + z * fnc_rve[2]
                            periodic_origins.append([xx, yy, zz, x, y, z])
            for p in periodic_origins:
                new_origin = [p[0], p[1], p[2]]
                if new_origin[1] > -fnc_new_diameter / 2.0001 and new_origin[1] < fnc_rve[1] + fnc_new_diameter / 2.0001 and new_origin[2] < fnc_rve[
                    2] + fnc_new_diameter / 2.0001 and new_origin[2] > -fnc_new_diameter / 2.0001 and new_origin[0] > -fnc_new_length / 2.0001 and new_origin[0] < fnc_rve[0] + fnc_new_length / 2.0001:
                    final_periodic_origins.append(p)

            return final_periodic_origins
        else:
            return []

    elif fnc_fiber_shape == 'sphere':
        if fnc_new_origin[0] > fnc_rve[0]-fnc_new_diameter/2.0001 or fnc_new_origin[0] < fnc_new_diameter/2.0001 or fnc_new_origin[1] > fnc_rve[1]-fnc_new_diameter/2.0001 or fnc_new_origin[1] < fnc_new_diameter/2.0001 or fnc_new_origin[2] > fnc_rve[2]-fnc_new_diameter/2.0001 or fnc_new_origin[2] < fnc_new_diameter/2.0001:
            for x in periodic_possibilities:
                for y in periodic_possibilities:
                    for z in periodic_possibilities:
                        if x == 0 and y == 0 and z == 0:
                            continue
                        else:
                            xx = fnc_new_origin[0] + x * fnc_rve[0]
                            yy = fnc_new_origin[1] + y * fnc_rve[1]
                            zz = fnc_new_origin[2] + z * fnc_rve[2]
                            periodic_origins.append([xx, yy, zz, x, y, z])

            for p in periodic_origins:
                new_origin = [p[0], p[1], p[2]]
                if new_origin[0] > -fnc_new_diameter / 2.0001 and new_origin[0] < fnc_rve[0] + fnc_new_diameter / 2.0001 and new_origin[1] > -fnc_new_diameter / 2.0001 and new_origin[1] < fnc_rve[1] + fnc_new_diameter / 2.0001 and new_origin[2] < fnc_rve[
                    2] + fnc_new_diameter / 2.0001 and new_origin[2] > -fnc_new_diameter / 2.0001:
                    final_periodic_origins.append(p)

            return final_periodic_origins

        else:
            return []
    # else:
    #
    #     for x in periodic_possibilities:
    #         for y in periodic_possibilities:
    #             for z in periodic_possibilities:
    #                 if x == 0 and y == 0 and z == 0:
    #                     continue
    #                 else:
    #                     xx = fnc_new_origin[0] + x * fnc_rve[0]
    #                     yy = fnc_new_origin[1] + y * fnc_rve[1]
    #                     zz = fnc_new_origin[2] + z * fnc_rve[2]
    #                     periodic_origins.append([xx, yy, zz, x, y, z])





#-------------------------------MAIN RSA Functions-----------------------------------#

# Uses gmsh boolean operations to detect collision/overlaps. (Slower than new_rsa)
def rsa(rve_info, fiber_collision_tolerance):


    rsa_attempts = 200
    rsa_attempts_per_fiber = 2000

    # Calculate the number of fibers required to achieve FVF
    rve_size = rve_info['MATRIX']['size']
    rve_volume = rve_size[0]*rve_size[1]*rve_size[2]

    for r in rve_info.keys():
        if 'COH' in r:
            continue
        elif 'UD' in r:
            volume = (rve_info[r]['dia'][0] ** 2) * np.pi * rve_size[0] / 4
        elif 'SFR' in r:
            volume = (rve_info[r]['dia'][0] ** 2) * np.pi * rve_info[r]['len'][0] / 4
        elif 'PR' in r:
            volume = ((rve_info[r]['dia'][0] / 2) ** 3) * np.pi * 4 / 3
        else:
            continue

        rve_info[r]['volume'] = volume
        rve_info[r]['numbers'] = int(rve_volume * rve_info[r]['FVC'] / volume)
        # print(rve_volume)
        # print(rve_info[r]['FVC'])
        # print(volume)
        print(f'For inclusion {r} required numbers are: ' , rve_info[r]['numbers'])
        rve_info[r]['fiber_dia_list'] = np.random.normal(rve_info[r]['dia'][0], rve_info[r]['dia'][1], rve_info[r]['numbers'])


        rve_info[r]['fiber_theta_list'] = np.random.normal(rve_info[r]['ori'][0], rve_info[r]['ori'][1],  rve_info[r]['numbers'])
        rve_info[r]['fiber_phi_list'] = np.random.normal(rve_info[r]['ori'][2], rve_info[r]['ori'][3], rve_info[r]['numbers'])
        rve_info[r]['fiber_length_list'] = np.random.normal(rve_info[r]['len'][0], rve_info[r]['len'][1], rve_info[r]['numbers'])




    # Total RSA Attempts
    RVE_achieved = False
    for r_a in range(1,rsa_attempts):

        if RVE_achieved is True:
            #print('RVE achieved')
            break

        # print('RSA Attempt #{}'.format(r_a))
        if r_a == rsa_attempts-1:
            print('ERROR. CANT MAKE RVE')
            break

        all_fiber_info = dict()
        cohesives = False
        coh_thickness = 0
        region = 10
        for r in rve_info.keys():
            fiber_info = dict()
            if 'MATRIX' in r or 'COH' in r:
                continue

            if any('COH-' + r in xx for xx in rve_info.keys()):
                cohesives = True
                coh_thickness = rve_info['COH-' + r]['thickness']

            for x in range(0, rve_info[r]['numbers']):
                fiber_info[str(x)] = [[-1, -1, -1], rve_info[r]['fiber_dia_list'][x], rve_info[r]['fiber_length_list'][x], rve_info[r]['fiber_theta_list'][x],
                                      rve_info[r]['fiber_phi_list'][x], r, [], 'no_periodicity']
            #print('Attempt #', r_a)
            # Loop through each fiber
            fail = False
            for f in range(0, rve_info[r]['numbers']):
                print('placing fiber #', f)
                if fail == True:
                    break
                for r_a_p_f in range(1, rsa_attempts_per_fiber):
                    if r_a_p_f == rsa_attempts_per_fiber-1:
                        fail = True
                        break
                    new_diameter = fiber_info[str(f)][1] + coh_thickness*2
                    if 'SFR' in r:
                        fiber_shape = 'cylinder'
                        new_direction_vector = np.array(
                            [np.sin(fiber_info[str(f)][4]) * np.cos(fiber_info[str(f)][3]), np.sin(fiber_info[str(f)][4]) *
                             np.sin(fiber_info[str(f)][3]), np.cos(fiber_info[str(f)][4])])
                        new_length = fiber_info[str(f)][2]
                        mapping_array_x = [-new_diameter / 3, rve_size[0] + new_diameter / 3]
                        mapping_array_y = [-new_diameter / 3, rve_size[1] + new_diameter / 3]
                        mapping_array_z = [-new_length / 3, rve_size[2] + new_length / 3]
                    elif 'UD' in r:
                        fiber_shape = 'UD'
                        new_direction_vector = np.array([1,0,0])
                        new_length = rve_size[0]*2
                        mapping_array_x = [rve_size[0]/2, rve_size[0]/2 ]
                        mapping_array_y = [-new_diameter / 3, rve_size[1] + new_diameter / 3]
                        mapping_array_z = [-new_diameter / 3, rve_size[2] + new_diameter / 3]


                        # mapping_array_y = [(rve_size[1]/2 - (region)/100*rve_size[1]/2 - new_diameter / 3, rve_size[1]/2 - (region-10)/100*rve_size[1]/2 - new_diameter / 3), (rve_size[1]/2 + region/100*rve_size[1]/2 + new_diameter / 3,rve_size[1]/2 + (region+10)/100*rve_size[1]/2 + new_diameter / 3)]
                        # mapping_array_z = [(rve_size[2]/2 - (region)/100*rve_size[2]/2 - new_diameter / 3, rve_size[2]/2 - (region-10)/100*rve_size[2]/2 - new_diameter / 3), (rve_size[2]/2 + region/100*rve_size[2]/2 + new_diameter / 3,rve_size[2]/2 + (region+10)/100*rve_size[2]/2 + new_diameter / 3)]
                    elif 'PR' in r:
                        fiber_shape = 'sphere'
                        new_direction_vector = np.array([1,0,0]) # Not needed
                        new_length = rve_size[0]*1.1 # Not needed
                        mapping_array_x = [-new_diameter / 3, rve_size[0] + new_diameter / 3]
                        mapping_array_y = [-new_diameter / 3, rve_size[1] + new_diameter / 3]
                        mapping_array_z = [-new_diameter / 3, rve_size[1] + new_diameter / 3]

                    # Generate random coordinates
                    mapping_array = [0,1]
                    new_origin = [np.interp(rnd.random(),mapping_array,mapping_array_x),
                                  np.interp(rnd.random(),mapping_array,mapping_array_y),
                                  np.interp(rnd.random(),mapping_array,mapping_array_z)]

                    #if check_collision(fiber_info.values(), new_origin, new_direction_vector, new_length, new_diameter) == True:
                    #    continue
                    if check_collision_gmsh_new(new_origin, new_direction_vector, new_length, new_diameter,fiber_shape,fiber_info,rve_info) == True:
                        continue
                    else:
                        p = check_periodicity_new(new_origin, new_direction_vector, new_length, new_diameter, rve_size, fiber_shape)


                        if p==False:
                            continue
                        elif p == []:
                            pass
                        else:
                            checked_periodics = 0
                            periodic_clash = False
                            for pp in p:
                                if check_collision_gmsh_new(pp[0:3], new_direction_vector, new_length, new_diameter,fiber_shape,fiber_info,rve_info) == True:
                                    periodic_clash = True
                                    break
                                else:
                                    checked_periodics += 1

                            if periodic_clash == True:
                                # clean_collision_check(checked_periodics)
                                continue

                        fiber_info[str(f)][0] = new_origin
                        if p == []:
                            break # Change
                        else:
                            fiber_info[str(f)][-1] = 'yes_periodicity'
                            for pp in p:
                                fiber_info[str(f)][5].append(pp)
                                fiber_info['p'+str(f) +','+ str(pp[3])+','+ str(pp[4])+','+ str(pp[5])] = [pp[0:3], fiber_info[str(f)][1], fiber_info[str(f)][2], fiber_info[str(f)][3], fiber_info[str(f)][4],
                                                                                            [], 'periodic_counter_part_of' + str(f)]
                            break

            if fail:
                break
            else:
                all_fiber_info[r] = fiber_info
                rve_info[r]['fiber_info'] = fiber_info
        if fail == False:
            RVE_achieved = True
    return


#-------------------------------Generate mesh-----------------------------------#
# def remove_volume_elsets(inp_in, inp_out):
#     with open(inp_in, "r", encoding="utf-8", errors="ignore") as f:
#         lines = f.readlines()
#     skipping = False
#     checking = True
#     with open(inp_out, "w") as f:
#         for l in lines:
#             if checking:
#                 if '*ELSET,ELSET=Volume' in l:
#                     skipping = True
#                 elif '*ELSET,ELSET=MATRIX' in l:
#                     skipping = False
#                     checking = False
#
#             if skipping:
#                 continue
#             else:
#                 f.write(l)



def create_periodic_mesh(rve_info, mesh_size, file_name_for_mesh,show_mesh,file_name_for_coords):
    rve_size = rve_info['MATRIX']['size']
    gmsh.model.setCurrent('final')
    final_box = gmsh.model.occ.addBox(0, 0, 0, rve_size[0], rve_size[1], rve_size[2])

    cohesives = False
    coh_thickness = 0
    for r in rve_info.keys():
        fiber_tags = []
        cohesive_tags = []
        cohesive_tags_2 = []
        if 'MATRIX' in r or 'COH' in r:
            continue

        fiber_info = rve_info[r]['fiber_info']
        if any('COH-' + r in xx for xx in rve_info.keys()):
            cohesives = True
            coh_thickness = rve_info['COH-' + r]['thickness']
        for a in fiber_info.values():
            if 'SFR' in r:
                direction = [np.sin(a[4]) * np.cos(a[3]), np.sin(a[4]) * np.sin(a[3]), np.cos(a[4])]
                final_fiber = gmsh.model.occ.addCylinder(a[0][0] - a[2] / 2 * direction[0],
                                                         a[0][1] - a[2] / 2 * direction[1],
                                                         a[0][2] - a[2] / 2 * direction[2],
                                                         a[2] * direction[0],
                                                         a[2] * direction[1],
                                                         a[2] * direction[2], a[1] / 2)
            elif 'UD' in r:
                direction = [1,0,0]
                if cohesives:
                    final_cohesive = gmsh.model.occ.addCylinder(a[0][0] - rve_size[0] * direction[0],
                                                             a[0][1] - rve_size[0]  * direction[1],
                                                             a[0][2] - rve_size[0]  * direction[2],
                                                             2*rve_size[0] * direction[0],
                                                             2*rve_size[0] * direction[1],
                                                             2*rve_size[0] * direction[2], a[1] / 2)

                final_fiber = gmsh.model.occ.addCylinder(a[0][0] - rve_size[0] * direction[0],
                                                         a[0][1] - rve_size[0]  * direction[1],
                                                         a[0][2] - rve_size[0]  * direction[2],
                                                         2*rve_size[0] * direction[0],
                                                         2*rve_size[0] * direction[1],
                                                         2*rve_size[0] * direction[2], -coh_thickness + a[1] / 2)




            elif 'PR' in r:
                final_fiber = gmsh.model.occ.addSphere(a[0][0], a[0][1], a[0][2], a[1] / 2)

            if cohesives:
                dsa = gmsh.model.occ.intersect([(3, final_cohesive)], [(3, final_box)], -1, True, False)



            asd = gmsh.model.occ.intersect([(3, final_fiber)], [(3, final_box)], -1, True, False)
            fiber_tags.append(asd[0])

            # print(dsa[0])
            # print(asd[0])
            if cohesives:
                gmsh.model.occ.cut([(3, 1)], dsa[0], -1, True, False)
                gmsh.model.occ.cut(dsa[0], asd[0], -1, True, False)
                cohesive_tags_2.extend([t for (d, t) in dsa[0] if d == 3])
                cohesive_tags.append(dsa[0])

            else:
                gmsh.model.occ.cut([(3, 1)], asd[0], -1, True, False)


        rve_info[r]['fiber_tags'] = fiber_tags
        if cohesives:
            rve_info['COH-' + r]['fiber_tags'] = cohesive_tags
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)


    #apply_periodic(rve_size[0], rve_size[1], rve_size[2])
    tol = 0.0001
    z0_face = gmsh.model.occ.getEntitiesInBoundingBox(-tol, -tol, -tol, rve_size[0] + tol, rve_size[1] + tol, tol,
                                                      2)
    z1_face = gmsh.model.occ.getEntitiesInBoundingBox(-tol, -tol, rve_size[2] - tol, rve_size[0] + tol,
                                                      rve_size[1] + tol, rve_size[2] + tol, 2)

    x0_face = gmsh.model.occ.getEntitiesInBoundingBox(-tol, -tol, -tol, tol, rve_size[1] + tol, rve_size[2] + tol,
                                                      2)
    x1_face = gmsh.model.occ.getEntitiesInBoundingBox(rve_size[0] - tol, -tol, -tol, rve_size[0] + tol,
                                                      rve_size[1] + tol, rve_size[2] + tol, 2)

    y0_face = gmsh.model.occ.getEntitiesInBoundingBox(-tol, -tol, -tol, rve_size[0] + tol, tol, rve_size[2] + tol,
                                                      2)
    y1_face = gmsh.model.occ.getEntitiesInBoundingBox(-tol, rve_size[1] - tol, -tol, rve_size[0] + tol,
                                                      rve_size[1] + tol, rve_size[2] + tol, 2)

    gmsh.model.occ.synchronize()






    # Set cohesives to be transfinite
    # pf = 0
    # pc = 0
    # cf = 0
    # cc = 0
    # both_surfs = [x0_face,x1_face]
    #
    # for surfs in both_surfs:
    #     for surf in surfs:
    #         gmsh.model.occ.synchronize()
    #         curves = gmsh.model.occ.getCurveLoops(surf[1])
    #
    #         if len(curves[0]) == 1:
    #             if len(curves[1][0]) == 1:
    #                 pf += 1
    #                 # print('perfect fiber')
    #             if len(curves[1][0]) == 2 or len(curves[1][0]) == 3:
    #                 cf += 1
    #                 # print('cutted fiber')
    #             if len(curves[1][0]) == 4 or len(curves[1][0]) == 6:
    #                 cc += 1
    #                 # print('cutted cohesive')
    #
    #
    #         elif len(curves[0]) == 2:
    #             if len(curves[1][0]) == 1 and len(curves[1][1]) == 1:
    #                 pc += 1


        # print(surf[1])
        # print(curves)





    gmsh.model.occ.synchronize()
    tol = 0.01
    # apply_periodic(rve_size[0], rve_size[1], rve_size[2])
    found_periodic = 0
    target_periodic = int(len(x0_face)) + int(len(y0_face)) + int(len(z0_face))
    for facex0 in x0_face:
        for facex1 in x1_face:
            facex0_center = gmsh.model.occ.getCenterOfMass(2, facex0[1])
            facex1_center = gmsh.model.occ.getCenterOfMass(2, facex1[1])
            gmsh.model.occ.synchronize()
            if abs((facex0_center[1]) - (facex1_center[1]))<tol and abs((facex0_center[2]) - (facex1_center[2]))<tol:
                gmsh.model.mesh.setPeriodic(2, [facex1[1]], [facex0[1]],
                                            [1, 0, 0, rve_size[0], 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
                x1_face.remove(facex1)
                found_periodic += 1
                break




    for facey0 in y0_face:
        for facey1 in y1_face:
            facey0_center = gmsh.model.occ.getCenterOfMass(2, facey0[1])
            facey1_center = gmsh.model.occ.getCenterOfMass(2, facey1[1])
            gmsh.model.occ.synchronize()
            if abs((facey0_center[0]) - (facey1_center[0]))<tol and abs((facey0_center[2]) - (facey1_center[2]))<tol:
                gmsh.model.mesh.setPeriodic(2, [facey1[1]], [facey0[1]],
                                            [1, 0, 0, 0, 0, 1, 0, rve_size[1], 0, 0, 1, 0, 0, 0, 0, 1])
                y1_face.remove(facey1)
                found_periodic += 1
                break


    for facez0 in z0_face:
        for facez1 in z1_face:
            facez0_center = gmsh.model.occ.getCenterOfMass(2, facez0[1])
            facez1_center = gmsh.model.occ.getCenterOfMass(2, facez1[1])
            gmsh.model.occ.synchronize()
            if abs((facez0_center[1]) - (facez1_center[1]))<tol and abs((facez0_center[0]) - (facez1_center[0]))<tol:
                gmsh.model.mesh.setPeriodic(2, [facez1[1]], [facez0[1]],
                                            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, rve_size[2], 0, 0, 0, 1])
                z1_face.remove(facez1)
                found_periodic += 1
                break


    if found_periodic!=target_periodic:
        raise('Could not find periodic mesh')

    gmsh.model.addPhysicalGroup(3, [final_box], name="MATRIX")
    for r in rve_info.keys():
        if 'MATRIX' in r:
            continue
        xxx = 0
        for aaa in rve_info[r]['fiber_tags']:
            gmsh.model.addPhysicalGroup(3, [aaa[0][1]], name=r + "_" + str(xxx))
            xxx+=1





    #################################################################################################
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.SaveGroupsOfElements", 1)  # Physical groups -> *ELSET
    gmsh.option.setNumber("Mesh.SaveAll", 0)


    if show_mesh:
        gmsh.fltk.run()
    #gmsh.finalize()
    gmsh.write(str(file_name_for_mesh))
    #remove_volume_elsets(str(file_name_for_mesh), str(file_name_for_mesh))
    # write_coh_coords(file_name_for_coords, rve_info)


    with open(str(file_name_for_mesh), 'r') as f:
        lines = f.readlines()

    lines = lines[2:]
    with open(str(file_name_for_mesh), 'w') as f:
        f.write('*PART, Name=RVE\n')
        for l in lines:
            f.write(l)
        f.write('*END PART\n')



#-------------------------------Write FEAP Files-----------------------------------#
def write_coh_coords(file_name_for_coords, rve_info):

    with open(file_name_for_coords, 'w') as f:
        for r in rve_info.keys():
            if 'COH-UD' in r:
                k = r.split('-')[-1]
                for a,b in zip(rve_info[k]['fiber_info'].keys(), rve_info[k]['fiber_info'].values()):
                    f.write(f'{a} {b[0][1]} {b[0][2]}\n')


def convert_to_feap(in_file, out_file, material_identifiers):
    with open(in_file, "r") as infile:
        lines = infile.readlines()[:-1]
    elements = dict()
    for m in material_identifiers:
        print(m)
        elements[m[1]] = []

    reading = 'none'
    change=False
    for l in lines:
        for m in material_identifiers:
            if '*ELSET,ELSET=' + m[0] in l:
                reading = m[1]
                change=True
        if change:
            change = False
            continue
        if reading != 'none':
            elmts = l.split(',')
            elmts = elmts[:-1]
            elmts = [int(x) for x in elmts]
            elements[reading] = elements[reading] + elmts


    with open(out_file, "w") as outfile:
        reading_nodes = False
        reading_elements = False
        for line in lines:

            # Example condition — customize as needed
            if "*NODE" in line:
                outfile.write('COORdinates\n')
                reading_nodes = True
                continue
            elif '******* E L E M E N T S *************' in line:
                #outfile.write('\n')
                reading_nodes = False
                reading_elements = True
                skip1 = True
                continue

            elif '*ELSET' in line:
                break

            if reading_nodes == True:
                x = line.split(',')
                outfile.write(x[0] + ' 0 ' + str(round(float(x[1]),20)) + ' '+ str(round(float(x[2]),20)) + ' ' + str(round(float(x[3]),20))+ '\n')

            if reading_elements == True:
                if '*ELEMENT' in line:
                    outfile.write('\nELEMents\n')
                    continue
                else:
                    x = line.split(',')
                    for m in material_identifiers:
                        if int(x[0]) in elements[m[1]]:
                            mat = m[1]
                            break
                    outfile.write(x[0] + ' 0' + mat + x[1]+ x[2]+ x[3]+ x[4])


def write_feap_input(mesh_file,new_folder,imn_validation_folder,rve_size,strain,steps,test_mesh_size,IMN_material,stage,r,g_id, IMN):
    files = ['DNS', 'IMN']
    new_folder.mkdir(exist_ok=True)
    for f in files:
        for x in range(1,7):
            nn = f'I_val_{f}_stage_{stage}_' + str(x)
            input_file = new_folder / nn
            with open(input_file, "w") as infile:
                infile.write('FEAP * * 4-Element Patch Test\n8,1,1,3,3,8\n \n')
                infile.write(f'PARA\nx={rve_size[0]}\ny={rve_size[1]}\nz={rve_size[2]}\na={test_mesh_size[0]}\nb={test_mesh_size[1]}\nc={test_mesh_size[2]}\n\n')


                material_num = 1
                first = True
                for m in IMN_material.values():
                    if f=='DNS':
                        infile.write(f'Material,{material_num}'  + '\n')
                        infile.write('Solid\n')
                        if m[0] == 1:
                            infile.write(f'Elas isot {m[1]} {m[2]}\n\n')
                        elif m[0] == 2:
                            infile.write(f'Elas isot {m[1]} {m[2]}\n')
                            infile.write(f'Plas mises {m[3]} {m[4]} {m[5]}\n')
                            infile.write(f'plas hard {m[6]}\n\n')
                        elif m[0] == 3:
                            infile.write(f'Elas neoh {m[1]} {m[2]} {m[3]}\n\n')
                        elif m[0] == 5:
                            infile.write(f'Elas isot {m[1]} {0.33}\n')
                        material_num += 1

                    else:
                        if first:
                            if 'IMN' in IMN:
                                infile.write(f'Mate,1\nSoli\nucon imn1 {int(str(imn_validation_folder)[-4:])} {len(list(IMN_material.values()))}\n')
                            else:
                                infile.write(f'Mate,1\nSoli\nucon dmn1 {int(str(imn_validation_folder)[-4:])} {len(list(IMN_material.values()))}\n')
                            if m[0] == 3:
                                infile.write('FINITE\n')
                            first = False
                        if m[0] == 1:
                            infile.write(f'Elas isot {material_num} {m[1]} {m[2]}\n')
                        elif m[0] == 2:
                            infile.write(f'Elas isot {material_num} {m[1]} {m[2]}\n')
                            infile.write(f'Plas mises {material_num} {m[3]} {m[4]} {m[5]}\n')
                            infile.write(f'plas hard {material_num} {m[6]}\n')
                        elif m[0] == 3:
                            infile.write(f'Elas neoh {material_num} {m[1]} {m[2]} {m[3]}\n')
                        elif m[0] == 5:
                            infile.write(f'Elas cohe {material_num} {m[1]} {m[2]}\n')
                        material_num += 1

                infile.write('\n')

                if f == 'DNS':
                    with open(mesh_file, "r") as meshfile:
                        mesh_lines = meshfile.read()
                        infile.write(mesh_lines)
                        infile.write("\n")
                else:
                    infile.write('SNOD\n1 0 0 0\n2 x 0 0\n3 x y 0\n4 0 y 0\n5 0 0 z\n6 x 0 z\n7 x y z\n8 0 y z\n\n')
                    infile.write('Blen\nSoli,a,b,c\nBric,8\nMATE,1\n1 2 3 4 5 6 7 8\n\n')


                # if x == 1:
                #     infile.write(f'Eboun\n1 {rve_size[0]} 1 0 0\n1 0 1 0 0\n\nEdisp\n1 {rve_size[0]} {strain * rve_size[0]} 0 0\n\nEND\n\n')
                # elif x == 2:
                #     infile.write(f'Eboun\n2 {rve_size[1]} 0 1 0\n2 0 0 1 0\n\nEdisp\n2 {rve_size[1]} 0 {strain * rve_size[1]} 0\n\nEND\n\n')
                # elif x == 3:
                #     infile.write(f'Eboun\n3 {rve_size[2]} 0 0 1\n3 0 0 0 1\n\nEdisp\n3 {rve_size[2]} 0 0 {strain * rve_size[2]}\n\nEND\n\n')
                # elif x == 4:
                #     infile.write(f'Eboun\n1 {rve_size[0]} 0 1 1\n1 0 1 1 1\n\nEdisp\n1 {rve_size[0]} 0 {strain * rve_size[0]} 0\n\nEND\n\n')
                # elif x == 5:
                #     infile.write(f'Eboun\n2 {rve_size[1]} 1 0 1\n2 0 1 1 1\n\nEdisp\n2 {rve_size[1]} 0 0 {2* strain * rve_size[1]}\n\nEND\n\n')
                # elif x == 6:
                #     infile.write(f'Eboun\n1 {rve_size[0]} 0 1 1\n1 0 1 1 1\n\nEdisp\n1 {rve_size[0]} 0 0 {strain * rve_size[0]}\n\nEND\n\n') # Check


                # HILL BC
                if x == 1:
                    infile.write(f'Eboun\n1 {rve_size[0]} 1 0 0\n1 0 1 0 0\n2 {rve_size[1]} 0 1 0\n2 0 0 1 0\n3 {rve_size[2]} 0 0 1\n3 0 0 0 1\n\nPeriodic Hill\nMech\n{strain} 0 0\n0 0 0\n0 0 0\n\nEND\n\n')
                elif x == 2:
                    infile.write(f'Eboun\n1 {rve_size[0]} 1 0 0\n1 0 1 0 0\n2 {rve_size[1]} 0 1 0\n2 0 0 1 0\n3 {rve_size[2]} 0 0 1\n3 0 0 0 1\n\nPeriodic Hill\nMech\n0 0 0\n0 {strain} 0\n0 0 0\n\nEND\n\n')
                elif x == 3:
                    infile.write(f'Eboun\n1 {rve_size[0]} 1 0 0\n1 0 1 0 0\n2 {rve_size[1]} 0 1 0\n2 0 0 1 0\n3 {rve_size[2]} 0 0 1\n3 0 0 0 1\n\nPeriodic Hill\nMech\n0 0 0\n0 0 0\n0 0 {strain}\n\nEND\n\n')
                elif x == 4:
                    infile.write(f'Eboun\n1 {rve_size[0]} 1 1 0\n1 0 1 1 0\n2 {rve_size[1]} 1 1 0\n2 0 1 1 0\n\nPeriodic Hill\nMech\n0 {strain/2} 0\n{strain/2} 0 0\n0 0 0\n\nEND\n\n')
                elif x == 6:
                    infile.write(f'Eboun\n1 {rve_size[0]} 1 0 1\n1 0 1 0 1\n3 {rve_size[2]} 1 0 1\n3 0 1 0 1\n\nPeriodic Hill\nMech\n0 0 {strain/2}\n0 0 0\n{strain/2} 0 0\n\nEND\n\n')
                elif x == 5:
                    infile.write(f'Eboun\n2 {rve_size[1]} 0 1 1\n2 0 0 1 1\n3 {rve_size[2]} 0 1 1\n3 0 0 1 1\n\nPeriodic Hill\nMech\n0 0 0\n0 0 {strain/2}\n0 {strain/2} 0\n\nEND\n\n')  # Check



                infile.write(f'PARA\ntt=100\ndt={int(100 / steps)}\nns=tt/dt\n\n')

                # Cyclic
                infile.write('BATCH\n  prop,,1\nEND\n2 1\n')
                infile.write('0, 0\n')
                # infile.write('30, 0.6\n')
                # infile.write('60, 0\n')
                infile.write('100, 1\n')
                infile.write('\n')

                # Linear
                # for ss in range(0,steps+1):
                #     infile.write(f'{ss*100/steps}, {ss/steps}\n')
                # infile.write('\n\n')


                infile.write('BATCh\nCHEC,mesh\nDT,,dt\nTOL,,1.d-18 1.d-10\nstre, aver\nLOOP,timestep,ns\nTIME\nTANGent,,1\nSOLVe\nStre, AVER\nNEXT,timestep\nEND\n\nStop')



# ---------------------------------------- Functions called by MAIN -------------------------------- #
def create_material_dict(rve_info, total_samples, training_dataset_folder,stage, mesh_per_config):


    training_dataset_folder.mkdir(exist_ok=True)
    # Create material data set
    ranges = {
        "E11": RangeSpec(10, 1500, "loguniform"),
        "E22": RangeSpec(10, 1500, "loguniform"),
        "E33": RangeSpec(10, 1500, "loguniform"),

        "nu12": RangeSpec(0.0, 0.49),
        "nu23": RangeSpec(0.0, 0.49),
        "nu31": RangeSpec(0.0, 0.49),

        "G12": RangeSpec(1, 1250, "loguniform"),
        "G23": RangeSpec(1, 1250, "loguniform"),
        "G31": RangeSpec(1, 1250, "loguniform"),
    }
    material_dictionary = {}
    material_samples = sample_many_orthotropic_materials(ranges, n_samples=total_samples * 10, seed=42)


    for i in range(0,len(material_samples),10):
        material_dictionary[str(int(i/10))] = {}
        material_dictionary[str(int(i/10))]['MATRIX'] = material_samples[i][1]
        material_dictionary[str(int(i/10))]['UD1'] = material_samples[i+1][1]
        material_dictionary[str(int(i/10))]['UD2'] = material_samples[i+2][1]
        material_dictionary[str(int(i / 10))]['UD3'] = material_samples[i + 3][1]
        material_dictionary[str(int(i/10))]['PR1'] = material_samples[i+4][1]
        material_dictionary[str(int(i/10))]['PR2'] = material_samples[i+5][1]
        material_dictionary[str(int(i / 10))]['PR3'] = material_samples[i + 6][1]
        material_dictionary[str(int(i/10))]['SFR1'] = material_samples[i+7][1]
        material_dictionary[str(int(i/10))]['SFR2'] = material_samples[i+8][1]
        material_dictionary[str(int(i / 10))]['SFR3'] = material_samples[i + 9][1]


    new_name = training_dataset_folder / f'material_dictionary.npz'
    np.savez(new_name, **material_dictionary, allow_pickle=True)


# Use analytical equations to detect collisions/overlaps
def new_rsa(rve_info, fiber_collision_tolerance):
    rsa_attempts = 50
    rsa_attempts_per_fiber =500000
    # Calculate the number of fibers required to achieve FVF
    rve_size = rve_info['MATRIX']['size']
    rve_volume = rve_size[0]*rve_size[1]*rve_size[2]

    # 3 types of inclusions implemented
    # UD: Unidirectional-fibers
    # SFR: Short-fibers reinforcement
    # PR: Particles reinforcement

    # COH: Cohesives (Not yet fully implemented)
    for r in rve_info.keys():
        if 'COH' in r:
            continue
        elif 'UD' in r:
            volume = (rve_info[r]['dia'][0] ** 2) * np.pi * rve_size[0] / 4
        elif 'SFR' in r:
            volume = (rve_info[r]['dia'][0] ** 2) * np.pi * rve_info[r]['len'][0] / 4
        elif 'PR' in r:
            volume = ((rve_info[r]['dia'][0] / 2) ** 3) * np.pi * 4 / 3
        else:
            continue

        rve_info[r]['volume'] = volume
        rve_info[r]['numbers'] = int(rve_volume * rve_info[r]['FVC'] / volume)
        print(f'For inclusion {r} required numbers are: ' , rve_info[r]['numbers'])



        rve_info[r]['fiber_dia_list'] = np.random.normal(rve_info[r]['dia'][0], rve_info[r]['dia'][1], rve_info[r]['numbers'])
        rve_info[r]['fiber_theta_list'] = np.random.normal(rve_info[r]['ori'][0], rve_info[r]['ori'][1],  rve_info[r]['numbers'])
        rve_info[r]['fiber_phi_list'] = np.random.normal(rve_info[r]['ori'][2], rve_info[r]['ori'][3], rve_info[r]['numbers'])
        rve_info[r]['fiber_length_list'] = np.random.normal(rve_info[r]['len'][0], rve_info[r]['len'][1], rve_info[r]['numbers'])




    # Total RSA Attempts
    RVE_achieved = False
    for r_a in range(1,rsa_attempts):

        if RVE_achieved is True:
            #print('RVE achieved')
            break

        # print('RSA Attempt #{}'.format(r_a))
        if r_a == rsa_attempts-1:
            print('ERROR. CANT MAKE RVE')
            break

        all_fiber_info = []

        cohesives = False
        coh_thickness = 0

        for r in rve_info.keys():
            fiber_info = dict()
            if 'MATRIX' in r or 'COH' in r:
                continue

            if any('COH-' + r in xx for xx in rve_info.keys()):
                cohesives = True
                coh_thickness = rve_info['COH-' + r]['thickness']

            for x in range(0, rve_info[r]['numbers']):
                fiber_info[str(x)] = [[-1, -1, -1], rve_info[r]['fiber_dia_list'][x], rve_info[r]['fiber_length_list'][x], rve_info[r]['fiber_theta_list'][x],
                                      rve_info[r]['fiber_phi_list'][x], r, [], 'no_periodicity']
            #print('Attempt #', r_a)
            # Loop through each fiber
            fail = False
            for f in range(0, rve_info[r]['numbers']):
                # print('Placing fiber #{}'.format(f))


                if fail == True:
                    break
                for r_a_p_f in range(1, rsa_attempts_per_fiber):

                    if r_a_p_f == rsa_attempts_per_fiber-1:
                        fail = True
                        break
                    new_diameter = fiber_info[str(f)][1] + coh_thickness*2
                    if 'SFR' in r:
                        fiber_shape = 'cylinder'
                        new_direction_vector = np.array(
                            [np.sin(fiber_info[str(f)][4]) * np.cos(fiber_info[str(f)][3]), np.sin(fiber_info[str(f)][4]) *
                             np.sin(fiber_info[str(f)][3]), np.cos(fiber_info[str(f)][4])])
                        new_length = fiber_info[str(f)][2]
                        mapping_array_x = [-new_length / 3, rve_size[0] + new_length / 3]
                        mapping_array_y = [-new_diameter / 3, rve_size[1] + new_diameter / 3]
                        mapping_array_z = [-new_diameter / 3, rve_size[2] + new_diameter / 3]

                        ## FIX ----------------------------------------------------

                    elif 'UD' in r:
                        fiber_shape = 'UD'
                        new_direction_vector = np.array([1,0,0])
                        new_length = rve_size[0]*2


                        mapping_array_x = [rve_size[0]/2, rve_size[0]/2 ]
                        mapping_array_y = [-new_diameter / 3, (rve_size[1] + new_diameter / 3)]
                        mapping_array_z = [-new_diameter / 3, (rve_size[2] + new_diameter / 3)]


                    elif 'PR' in r:
                        fiber_shape = 'sphere'
                        new_direction_vector = np.array([1,0,0]) # Not needed
                        new_length = rve_size[0]*1.1 # Not needed
                        mapping_array_x = [-new_diameter / 3, rve_size[0] + new_diameter / 3]
                        mapping_array_y = [-new_diameter / 3, rve_size[1] + new_diameter / 3]
                        mapping_array_z = [-new_diameter / 3, rve_size[1] + new_diameter / 3]

                    # Generate random coordinates
                    mapping_array = [0, 1]
                    new_origin = [np.interp(rnd.random(), mapping_array, mapping_array_x),
                                  np.interp(rnd.random(), mapping_array, mapping_array_y),
                                  np.interp(rnd.random(), mapping_array, mapping_array_z)]



                    if check_collision_gmsh_new(new_origin, new_direction_vector, new_length, new_diameter+fiber_collision_tolerance,fiber_shape,all_fiber_info,fiber_info,rve_info) == True:
                        continue
                    else:
                        p = check_periodicity_new(new_origin, new_direction_vector, new_length, new_diameter, rve_size, fiber_shape)

                        if p==False:
                            continue
                        elif p == []:
                            pass
                        else:
                            checked_periodics = 0
                            periodic_clash = False
                            for pp in p:
                                if check_collision_gmsh_new(pp[0:3], new_direction_vector, new_length, new_diameter+fiber_collision_tolerance,fiber_shape,all_fiber_info, fiber_info,rve_info) == True:
                                    periodic_clash = True
                                    break
                                else:
                                    checked_periodics += 1

                            if periodic_clash == True:
                                # clean_collision_check(checked_periodics)
                                continue

                        fiber_info[str(f)][0] = new_origin
                        if p == []:
                            break # Change
                        else:
                            fiber_info[str(f)][-1] = 'yes_periodicity'
                            for pp in p:
                                fiber_info[str(f)][6].append(pp)
                                fiber_info['p'+str(f) +','+ str(pp[3])+','+ str(pp[4])+','+ str(pp[5])] = [pp[0:3], fiber_info[str(f)][1], fiber_info[str(f)][2], fiber_info[str(f)][3], fiber_info[str(f)][4],r,
                                                                                            [], 'periodic_counter_part_of' + str(f)]
                            break

            if fail:
                break
            else:
                all_fiber_info.append(fiber_info)
                rve_info[r]['fiber_info'] = fiber_info
        if fail == False:
            RVE_achieved = True
    return


def cleanup(base_dir):
    import shutil
    prefix = "folder_"  # folders starting with this
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path) and name.startswith(prefix):
            shutil.rmtree(full_path)


def create_mesh(rve_info, mesh_size, fiber_collision_tolerance, mesh_folder, show_mesh,stage,r, graph_id):
    gmsh.finalize()
    gmsh.initialize()
    gmsh.model.add("collision_check")
    gmsh.model.add("periodicity_check")
    gmsh.model.add("final")
    gmsh.option.setNumber('General.Verbosity', 0)

    file_name_for_mesh = mesh_folder / f'gmsh_stage_{stage}_rve_{r}_mesh_{graph_id}.inp'  # From gmsh in abaqus format
    file_name_for_coords = mesh_folder / f'gmsh_stage_{stage}_rve_{r}_mesh_{graph_id}.txt'

    # rsa(rve_info, fiber_collision_tolerance)
    new_rsa(rve_info, fiber_collision_tolerance)
    create_periodic_mesh(rve_info, mesh_size, file_name_for_mesh,show_mesh,file_name_for_coords)



def create_abaqus_main_file(mesh_folder,stage,r, graph_id):


    import subprocess
    os.chdir(mesh_folder)
    # log_folder = Path(mesh_folder) / 'logs'
    log_folder = Path('logs')
    log_folder.mkdir(exist_ok=True)
    log = log_folder / f"create_main_abaqus_file_stage_{stage}_rve_{r}_mesh_{graph_id}.log"
    # os.system(f'abaqus cae noGUI=temp_abaqus.py -- {stage} {r} {graph_id}')
    with open(log, "w") as f:
        subprocess.run(
            f"abaqus cae noGUI=temp_abaqus.py -- {0} {stage} {r} {graph_id}",
            shell=True,
            stdout=f,
            stderr=f
        )

    # log = log_folder / f"create_abaqus_graph_file_stage_{stage}_rve_{r}_mesh_{graph_id}.log"
    # with open(log, "w") as f:
    #     subprocess.run(
    #         f"abaqus cae noGUI=temp_abaqus.py -- {1} {stage} {r} {graph_id}",
    #         shell=True,
    #         stdout=f,
    #         stderr=f
    #     )
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')

# def create_abaqus_main_file(mesh_folder, stage, r, graph_id):
#     import subprocess
#     from pathlib import Path
#     import os
#     mesh_folder = Path(mesh_folder).resolve()
#     old_cwd = Path.cwd()
#
#     log_folder = mesh_folder / 'logs'
#     log_folder.mkdir(exist_ok=True)
#
#     log = log_folder / f"create_main_abaqus_file_stage_{stage}_rve_{r}_mesh_{graph_id}.log"
#
#     try:
#         with open(log, "w") as f:
#             result = subprocess.run(
#                 f"abaqus cae noGUI=temp_abaqus.py -- 0 {stage} {r} {graph_id}",
#                 cwd=str(mesh_folder),
#                 shell=True,
#                 stdout=f,
#                 stderr=f
#             )
#
#         if result.returncode != 0:
#             raise RuntimeError(f"Abaqus main-file generation failed. Check log: {log}")
#
#         job_inp = mesh_folder / f"Job_stage_{stage}_rve_{r}_mesh_{graph_id}.inp"
#
#         if not job_inp.exists():
#             raise RuntimeError(f"Expected Job inp was not created: {job_inp}")
#
#         txt = job_inp.read_text(errors="ignore")
#
#         if "*Step" not in txt and "*STEP" not in txt:
#             raise RuntimeError(f"Generated Job inp has no *Step block: {job_inp}")
#
#         if "*End Step" not in txt and "*END STEP" not in txt:
#             raise RuntimeError(f"Generated Job inp has no *End Step block: {job_inp}")
#
#     finally:
#         os.chdir(str(old_cwd))

# def create_mesh_graph(mesh_folder,stage, r, graph_id):
#
#     import subprocess
#     os.chdir(mesh_folder)
#     # log_folder = Path(mesh_folder) / 'logs'
#     log_folder = Path('logs')
#     log_folder.mkdir(exist_ok=True)
#     log = log_folder / f"create_mesh_graph_stage_{stage}_rve_{r}_mesh_{graph_id}.log"
#     # os.system(f'abaqus cae noGUI=newGNN_graph.py -- {stage} {r} {graph_id}')
#     with open(log, "w") as f:
#         subprocess.run(
#             f"abaqus cae noGUI=Extracting_graph_from_mesh.py -- {stage} {r} {graph_id}",
#             shell=True,
#             stdout=f,
#             stderr=f
#         )
#
#     os.chdir('..')
#     os.chdir('..')
#     os.chdir('..')

def create_mesh_graph(mesh_folder, stage, r, graph_id, mode):
    import subprocess
    from pathlib import Path
    import os

    mesh_folder = Path(mesh_folder).resolve()
    old_cwd = Path.cwd()

    log_folder = mesh_folder / 'logs'
    log_folder.mkdir(exist_ok=True)

    log = log_folder / f"create_mesh_graph_stage_{stage}_rve_{r}_mesh_{graph_id}.log"

    try:
        with open(log, "w") as f:
            # if mode == 'GNN_IMN':
            result = subprocess.run(
                f"abaqus cae noGUI=Extracting_graph_from_mesh.py -- {stage} {r} {graph_id}",
                cwd=str(mesh_folder),
                shell=True,
                stdout=f,
                stderr=f
            )
            # elif mode == 'GNN_DMN':
            result = subprocess.run(
                f"abaqus cae noGUI=Extracting_graph_from_mesh_DMN.py -- {stage} {r} {graph_id}",
                cwd=str(mesh_folder),
                shell=True,
                stdout=f,
                stderr=f
            )
        if result.returncode != 0:
            raise RuntimeError(f"Mesh graph generation failed. Check log: {log}")

    finally:
        os.chdir(str(old_cwd))



def create_abaqus_input_files(
    rve_info, sample_num, samples,
    training_dataset_folder, mesh_folder,
    stage, r, graph_id, key_map
):
    import numpy as np

    material_dictionary = {}
    x = np.load(training_dataset_folder / 'material_dictionary.npz', allow_pickle=True)
    for k in x.keys():
        material_dictionary[k] = x[k].item()

    template_path = mesh_folder / f'Job_stage_{stage}_rve_{r}_mesh_{graph_id}.inp'

    with open(template_path, 'r') as f:
        data = f.readlines()

    template_txt = ''.join(data)
    if '*Step' not in template_txt and '*STEP' not in template_txt:
        raise RuntimeError(f'Template has no *Step block: {template_path}')

    phase_names = set(rve_info.keys())

    def parse_material_name(line):
        # Handles: *Material, name=MATRIX
        parts = line.strip().split(',')
        for p in parts:
            if p.strip().lower().startswith('name='):
                return p.split('=', 1)[1].strip()
        return None

    def anisotropic_elastic_block(C):
        vals = [
            C[0,0],
            C[0,1], C[1,1],
            C[0,2], C[1,2], C[2,2],
            C[0,3], C[1,3], C[2,3], C[3,3],
            C[0,4], C[1,4], C[2,4], C[3,4], C[4,4],
            C[0,5], C[1,5], C[2,5], C[3,5], C[4,5], C[5,5],
        ]

        out = ['*Elastic, type=ANISOTROPIC\n']
        for a in range(0, len(vals), 8):
            out.append(', '.join('%.16g' % float(v) for v in vals[a:a+8]) + '\n')
        return out

    def orthotropic_elastic_block(C):
        vals = [
            C[0, 0],
            C[0, 1], C[1, 1],
            C[0, 2], C[1, 2], C[2, 2],
            C[5, 5], C[4, 4],
            C[3, 3],
        ]

        out = ['*Elastic, type=ORTHOTROPIC\n']
        out.append(', '.join('%.16g' % float(v) for v in vals[:8]) + '\n')
        out.append('%.16g\n' % float(vals[8]))
        return out

    for i in range(sample_num, sample_num + samples):
        new_dir = training_dataset_folder / f'folder_{i}'
        new_dir.mkdir(exist_ok=True)

        new_name = new_dir / f'I_{i}_stage_{stage}_rve_{r}_mesh_{graph_id}.inp'

        key_map[str(i)] = {
            'ids': [stage, r, graph_id],
            'feilename': f'graph_stage_{stage}_rve_{r}_mesh_{graph_id}.npz',
            'phases': []
        }

        out_lines = []
        j = 0

        while j < len(data):
            line = data[j]
            stripped = line.strip()
            lower = stripped.lower()

            if lower.startswith('*material'):
                mat_name = parse_material_name(line)

                out_lines.append(line)
                j += 1

                # Collect everything inside this material block until next *Material
                material_subblock = []
                while j < len(data):
                    next_line = data[j]
                    next_lower = next_line.strip().lower()

                    if next_lower.startswith('*material'):
                        break

                    material_subblock.append(next_line)
                    j += 1

                if mat_name in phase_names:
                    if str(i) not in material_dictionary:
                        raise RuntimeError(f'Missing material dictionary entry for sample {i}')

                    if mat_name not in material_dictionary[str(i)]:
                        raise RuntimeError(f'Missing material {mat_name} for sample {i}')

                    key_map[str(i)]['phases'].append(mat_name)

                    C = material_dictionary[str(i)][mat_name]
                    # out_lines.extend(anisotropic_elastic_block(C))
                    out_lines.extend(orthotropic_elastic_block(C))

                    # Do not copy old elastic data
                    # But preserve non-elastic keywords inside material if any
                    k = 0
                    while k < len(material_subblock):
                        ml = material_subblock[k]
                        mlower = ml.strip().lower()

                        if mlower.startswith('*elastic'):
                            k += 1
                            while k < len(material_subblock):
                                if material_subblock[k].strip().startswith('*'):
                                    break
                                k += 1
                            continue

                        out_lines.append(ml)
                        k += 1

                else:
                    out_lines.extend(material_subblock)

                continue

            out_lines.append(line)
            j += 1

        with open(new_name, 'w') as f:
            f.writelines(out_lines)

        txt = ''.join(out_lines)

        if '*Step' not in txt and '*STEP' not in txt:
            raise RuntimeError(f'Generated file has no *Step block: {new_name}')

        if '*End Step' not in txt and '*END STEP' not in txt:
            raise RuntimeError(f'Generated file has no *End Step block: {new_name}')

# def create_abaqus_input_files(rve_info, sample_num, samples, training_dataset_folder,mesh_folder, stage, r, graph_id, key_map):
#
#
#     # base_dir = Path(F_Training_data_generation + training_data_folder_id)
#     # base_dir.mkdir(exist_ok=True)
#     material_dictionary = {}
#     x = np.load(training_dataset_folder / f'material_dictionary.npz', allow_pickle=True)
#     for k in x.keys():
#         material_dictionary[k] = x[k].item()
#
#
#     with open(mesh_folder / f'Job_stage_{stage}_rve_{r}_mesh_{graph_id}.inp', 'r') as file:
#         data = file.readlines()
#
#     # Copy the generated input files to separate folders and write the material data
#     for i in range(sample_num, samples+sample_num):
#         new_dir = training_dataset_folder / f'folder_{i}'
#         new_dir.mkdir(exist_ok=True)
#         key_map[str(i)] = {}
#         new_name = new_dir / f'I_{i}_stage_{stage}_rve_{r}_mesh_{graph_id}.inp'
#
#         key_map[str(i)]['ids'] = [stage,r,graph_id]
#         key_map[str(i)]['feilename'] = f'graph_stage_{stage}_rve_{r}_mesh_{graph_id}.npz'
#         key_map[str(i)]['phases'] = []
#         with open(new_name, 'w') as file:
#             skip = 0
#             for l in data:
#                 if skip==0:
#                     if '*Material, name=' in l:
#                         k = l.split('=')[-1][:-1]
#                         for kk in rve_info.keys():
#                             skip = 3
#                             if k in kk:
#                                 key_map[str(i)]['phases'].append(k)
#                                 file.write(l)
#                                 file.write('*Elastic, type=ORTHOTROPIC\n')
#                                 file.write(
#                                     f'{float(material_dictionary[str(i)][k][0, 0])}, {float(material_dictionary[str(i)][k][0, 1])}, {float(material_dictionary[str(i)][k][1, 1])},'
#                                     f'{float(material_dictionary[str(i)][k][0, 2])}, {float(material_dictionary[str(i)][k][1, 2])}, {float(material_dictionary[str(i)][k][2, 2])},'
#                                     f'{float(material_dictionary[str(i)][k][3, 3])},{float(material_dictionary[str(i)][k][4, 4])},\n{float(material_dictionary[str(i)][k][5, 5])}\n')
#                                 break
#                                 # file.write(f'{float(material_dictionary[k + '_stiff_mat'][i][0,0])}, {float(material_dictionary[k + '_stiff_mat'][i][0,1])}, {float(material_dictionary[k + '_stiff_mat'][i][1,1])},'
#                                 #            f'{float(material_dictionary[k + '_stiff_mat'][i][0,2])}, {float(material_dictionary[k + '_stiff_mat'][i][1,2])}, {float(material_dictionary[k + '_stiff_mat'][i][2,2])},'
#                                 #            f'{float(material_dictionary[k + '_stiff_mat'][i][3,3])},{float(material_dictionary[k + '_stiff_mat'][i][4,4])},\n{float(material_dictionary[k + '_stiff_mat'][i][5,5])}\n')
#                                 # break
#
#                     else:
#                         file.write(l)
#                 else:
#                     skip-=1


# def create_abaqus_input_files(
#     rve_info, sample_num, samples,
#     training_dataset_folder, mesh_folder,
#     stage, r, graph_id, key_map
# ):
#     material_dictionary = {}
#     x = np.load(training_dataset_folder / 'material_dictionary.npz', allow_pickle=True)
#     for k in x.keys():
#         material_dictionary[k] = x[k].item()
#
#     template_path = mesh_folder / f'Job_stage_{stage}_rve_{r}_mesh_{graph_id}.inp'
#
#     with open(template_path, 'r') as file:
#         data = file.readlines()
#
#     for i in range(sample_num, samples + sample_num):
#         new_dir = training_dataset_folder / f'folder_{i}'
#         new_dir.mkdir(exist_ok=True)
#
#         new_name = new_dir / f'I_{i}_stage_{stage}_rve_{r}_mesh_{graph_id}.inp'
#
#         key_map[str(i)] = {
#             'ids': [stage, r, graph_id],
#             'feilename': f'graph_stage_{stage}_rve_{r}_mesh_{graph_id}.npz',
#             'phases': []
#         }
#
#         with open(new_name, 'w') as file:
#             j = 0
#
#             while j < len(data):
#                 line = data[j]
#                 line_clean = line.strip()
#                 line_lower = line_clean.lower()
#
#                 if line_lower.startswith('*material') and 'name=' in line_lower:
#                     mat_name = line.split('=')[-1].strip()
#
#                     should_replace = False
#                     for kk in rve_info.keys():
#                         if mat_name in kk:
#                             should_replace = True
#                             break
#
#                     if should_replace:
#                         key_map[str(i)]['phases'].append(mat_name)
#
#                         file.write(line)
#
#                         C = material_dictionary[str(i)][mat_name]
#
#                         # IMPORTANT:
#                         # If C is a 6x6 stiffness matrix, use ANISOTROPIC, not ORTHOTROPIC.
#                         file.write('*Elastic, type=ANISOTROPIC\n')
#
#                         vals = [
#                             C[0,0],
#                             C[0,1], C[1,1],
#                             C[0,2], C[1,2], C[2,2],
#                             C[0,3], C[1,3], C[2,3], C[3,3],
#                             C[0,4], C[1,4], C[2,4], C[3,4], C[4,4],
#                             C[0,5], C[1,5], C[2,5], C[3,5], C[4,5], C[5,5],
#                         ]
#
#                         for a in range(0, len(vals), 8):
#                             file.write(', '.join('%.16g' % float(v) for v in vals[a:a+8]) + '\n')
#
#                         # Skip old material data until next keyword
#                         j += 1
#                         while j < len(data):
#                             s = data[j].strip()
#                             if s.startswith('*'):
#                                 break
#                             j += 1
#
#                         # If old next line is *Elastic, skip that block too
#                         if j < len(data) and data[j].strip().lower().startswith('*elastic'):
#                             j += 1
#                             while j < len(data):
#                                 s = data[j].strip()
#                                 if s.startswith('*'):
#                                     break
#                                 j += 1
#
#                         continue
#
#                     else:
#                         # Preserve material untouched
#                         file.write(line)
#                         j += 1
#                         continue
#
#                 file.write(line)
#                 j += 1
#
#         # Safety check
#         with open(new_name, 'r') as check_file:
#             txt = check_file.read()
#
#         if '*Step' not in txt and '*STEP' not in txt:
#             raise RuntimeError(f'Generated file has no *Step block: {new_name}')
#
#         if '*End Step' not in txt and '*END STEP' not in txt:
#             raise RuntimeError(f'Generated file has no *End Step block: {new_name}')

def solve_abaqus_input_files(samples, training_dataset_folder):
    # base_dir = Path(F_Training_data_generation + training_data_folder_id)
    # base_dir.mkdir(exist_ok=True)
    print('')
    extensions_to_delete = [
        ".com", ".prt", ".msg", ".sta", ".sim", ".res"
    ]
    import subprocess
    root = training_dataset_folder
    running = -1
    for root_dir, _, files in os.walk(root):
        for file in files:
            # if file.startswith('I_') and file.endswith(f'_{graph_id}.inp') and f'stage_{stage}' in file:
            if file.startswith('I_'):
                job = os.path.splitext(file)[0]
                running += 1
                print(f"\rRunning job: {job}     |    Completed={int(100*running/(samples))} %...", end ="")

                log_path = os.path.join(root_dir, f"{job}_run.log")
                with open(log_path, "w") as log:
                # with open(os.path.join(root_dir, f"{job}_run.log"), "w") as log:
                    result = subprocess.run(
                        f'abaqus job={job} input="{file}" interactive',
                        cwd=root_dir,
                        shell=True,
                        stdout=log,
                        stderr=log,
                        check=True
                    )
                    # ---- Delete unwanted files AFTER job finishes ----

                if result.returncode != 0:
                    raise RuntimeError(
                        f"\nAbaqus system-level failure in job {job}. "
                        f"Return code: {result.returncode}. Log: {log_path}"
                    )

                if abaqus_job_failed(root_dir, job):
                    raise RuntimeError(
                        f"\nAbaqus analysis failed in job {job}. "
                        f"Check: {log_path}, {job}.dat, {job}.msg, {job}.sta"
                    )

                for ext in extensions_to_delete:
                    path = os.path.join(root_dir, job + ext)
                    if os.path.exists(path):
                        os.remove(path)
                for ext in extensions_to_delete:
                    path = os.path.join(root_dir, job + ext)
                    if os.path.exists(path):
                        os.remove(path)

    print('')




def solve_abaqus_main_files(samples, training_dataset_folder):
    # base_dir = Path(F_Training_data_generation + training_data_folder_id)
    # base_dir.mkdir(exist_ok=True)
    print('')
    extensions_to_delete = [
        ".com", ".prt", ".msg", ".sta", ".sim", ".res"
    ]
    import subprocess
    root = training_dataset_folder / 'Meshes'
    running = -1
    for root_dir, _, files in os.walk(root):
        for file in files:
            # if file.startswith('I_') and file.endswith(f'_{graph_id}.inp') and f'stage_{stage}' in file:
            if file.startswith('Job_'):
                job = os.path.splitext(file)[0]
                running += 1
                print(f"\rRunning Main job: {job}     |    Completed={int(100*running/(samples))} %...", end ="")

                log_path = os.path.join(root_dir, f"{job}_run.log")
                with open(log_path, "w") as log:
                # with open(os.path.join(root_dir, f"{job}_run.log"), "w") as log:
                    result = subprocess.run(
                        f'abaqus job={job} input="{file}" interactive',
                        cwd=root_dir,
                        shell=True,
                        stdout=log,
                        stderr=log,
                        check=True
                    )
                    # ---- Delete unwanted files AFTER job finishes ----

                if result.returncode != 0:
                    raise RuntimeError(
                        f"\nAbaqus system-level failure in job {job}. "
                        f"Return code: {result.returncode}. Log: {log_path}"
                    )

                if abaqus_job_failed(root_dir, job):
                    raise RuntimeError(
                        f"\nAbaqus analysis failed in job {job}. "
                        f"Check: {log_path}, {job}.dat, {job}.msg, {job}.sta"
                    )

                for ext in extensions_to_delete:
                    path = os.path.join(root_dir, job + ext)
                    if os.path.exists(path):
                        os.remove(path)
                for ext in extensions_to_delete:
                    path = os.path.join(root_dir, job + ext)
                    if os.path.exists(path):
                        os.remove(path)

    print('')

def abaqus_job_failed(root_dir, job):
    files_to_check = [
        job + ".dat",
        # job + ".msg",
        # job + ".sta",
        # job + ".log",
    ]

    fatal_patterns = [
        "THE PROGRAM HAS DISCOVERED",
        "FATAL ERROR",
        "***ERROR",
        "EXECUTION IS TERMINATED",
        # "Abaqus Error",
        "Analysis Input File Processor exited with an error",
    ]

    success_patterns = [
        "ANALYSIS COMPLETE"
        "THE ANALYSIS HAS COMPLETED SUCCESSFULLY",
        "Abaqus JOB %s COMPLETED" % job,
    ]

    combined = ""

    for fname in files_to_check:
        path = os.path.join(root_dir, fname)
        if os.path.exists(path):
            with open(path, "r", errors="ignore") as f:
                combined += "\n" + f.read()

    upper = combined.upper()
    for pat in fatal_patterns:
        if pat.upper() in upper:
            return True

    # Optional stricter check:
    # if no success message is found, treat as failed
    # if not any(pat.upper() in upper for pat in success_patterns):
    #     return True

    return False


def create_FEAP_validation_files(rve_info, strain, mesh_folder, new_folder,imn_validation_folder, steps, test_mesh_size, IMN_material,stage, r,g_id, IMN):


    file_name_for_mesh = mesh_folder / f'gmsh_stage_{stage}_rve_{r}_mesh_{g_id}.inp'  # From gmsh in abaqus format
    file_name_for_FEAP_mesh = mesh_folder / f'val_IMN_stage_{stage}_rve_{r}_mesh_{g_id}'  # From abaqus to FEAP format
    material_identifiers = [[y[:-3], y[-3:]] for y in list(IMN_material.keys())]
    convert_to_feap(file_name_for_mesh, file_name_for_FEAP_mesh,material_identifiers)

    write_feap_input(file_name_for_FEAP_mesh,new_folder,imn_validation_folder,rve_info['MATRIX']['size'],strain,steps,test_mesh_size,IMN_material,stage, r,g_id, IMN)

















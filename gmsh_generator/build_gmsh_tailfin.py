"""Boundary-layer-explicit tailfin mesh via Gmsh Python API -> gmshToFoam.

Replaces snappyHexMesh for wall-resolved (y+<1) LSB work. Builds the fin by
lofting SD8020 sections (root/tip, sweep offset, TE blunted 0.1% chord),
cuts it out of the same farfield box as blockMeshDict, grows explicit prisms
from the fin walls with a BoundaryLayer field, and writes MSH2 ASCII for
gmshToFoam.

Usage:
  python3 gmsh_generator/build_gmsh_tailfin.py --sweep 10.0 --ar 3.362 \\
      --taper 0.40 --out runs/case_gmsh01/tailfin.msh
  # then convert (inside openfoam env):
  openfoam -c 'gmshToFoam tailfin.msh'   # run in case dir with system/controlDict etc.

Defaults target y+<1: hfirst=2.5e-05 m (cell centre ~12.5 um -> y+~0.2 avg,
LE peak ~0.6), ratio 1.25, total ~1.5 mm (~13 layers, ~50% of laminar BL).
"""
import argparse
import math
import os
import sys

try:
    import gmsh
except ImportError:
    sys.exit('pip install gmsh  (needs gmsh>=4.11 python API)')

# same box as openfoam_template/system/blockMeshDict
BOX = (-0.5, -0.5, -0.1, 2.0, 1.0, 0.6)  # xmin,ymin,zmin, dx,dy,dz


def load_airfoil(dat_path, chord, te_frac=0.001):
    pts = []
    with open(dat_path) as f:
        for line in f.readlines()[1:]:
            p = line.strip().split()
            if len(p) >= 2:
                pts.append((float(p[0]) * chord, float(p[1]) * chord))
    # blunt sharp TE: first+last points coincide at (chord, 0)
    gap = te_frac * chord
    if abs(pts[0][0] - pts[-1][0]) < 1e-12 and abs(pts[0][1] - pts[-1][1]) < 1e-12:
        pts[0] = (pts[0][0], pts[0][1] + gap / 2)
        pts[-1] = (pts[-1][0], pts[-1][1] - gap / 2)
    return pts


def section_surface(pts, z, dx):
    """Closed OCC plane surface of airfoil section shifted by (dx, 0, z)."""
    # split top / bottom at leading edge (min x)
    le = min(range(len(pts)), key=lambda i: pts[i][0])
    top = pts[:le + 1]
    bot = pts[le:]
    def chain_tags(chain):
        return [gmsh.model.occ.addPoint(x + dx, y, z) for x, y in chain]
    p_top = chain_tags(top)          # TE-top ... LE
    p_bot = chain_tags(bot[1:])      # ... TE-bottom (skip duplicate LE)
    c_top = gmsh.model.occ.addSpline(p_top)
    c_bot = gmsh.model.occ.addSpline([p_top[-1]] + p_bot)
    c_te = gmsh.model.occ.addLine(p_bot[-1], p_top[0])
    return gmsh.model.occ.addCurveLoop([c_top, c_bot, c_te])


def build(sweep_deg, ar, taper, dat_path, out_msh,
          hfirst=2.5e-05, ratio=1.25, bl_thickness=1.5e-03,
          lc_min=0.002, lc_max=0.05, dist_max=0.5):
    root = 0.15
    tip = root * taper
    span = ar * 0.5 * (root + tip)
    sweep_off = span * math.tan(math.radians(sweep_deg))

    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 1)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)  # gmshToFoam compat
    gmsh.model.add('tailfin')

    rp = load_airfoil(dat_path, root)
    tp = load_airfoil(dat_path, tip)
    wire_root = section_surface(rp, 0.0, 0.0)
    wire_tip = section_surface(tp, span, sweep_off)
    gmsh.model.occ.synchronize()
    gmsh.model.occ.addThruSections(
        [wire_root, wire_tip], makeSolid=True, makeRuled=True)
    gmsh.model.occ.synchronize()

    x0, y0, z0, dx, dy, dz = BOX
    box = gmsh.model.occ.addBox(x0, y0, z0, dx, dy, dz)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    fin_vol = [v for v in vols if v[1] != box]
    fluid = gmsh.model.occ.cut([(3, box)], fin_vol)
    gmsh.model.occ.synchronize()

    # classify boundary faces by bbox centre
    tol = 1e-6
    groups = {'inlet': [], 'outlet': [], 'sides': [], 'symmetry': [], 'tailfin': []}
    for dim, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        if abs(cx - x0) < tol:
            groups['inlet'].append(tag)
        elif abs(cx - (x0 + dx)) < tol:
            groups['outlet'].append(tag)
        elif abs(cz - z0) < tol:
            groups['symmetry'].append(tag)
        elif abs(cy - y0) < tol or abs(cy - (y0 + dy)) < tol or abs(cz - (z0 + dz)) < tol:
            groups['sides'].append(tag)
        else:
            groups['tailfin'].append(tag)
    assert groups['tailfin'], 'no fin walls found - classification failed'
    for name, tags in groups.items():
        assert tags, f'empty patch group: {name}'
        gmsh.model.addPhysicalGroup(2, tags, name=name)
    fluid_vols = [v[1] for v in gmsh.model.getEntities(3)]
    gmsh.model.addPhysicalGroup(3, fluid_vols, name='fluid')

    # --- mesh size: fine near fin, coarse far away ---
    d = gmsh.model.mesh.field.add('Distance', 1)
    gmsh.model.mesh.field.setNumbers(1, 'FacesList', groups['tailfin'])
    t = gmsh.model.mesh.field.add('Threshold', 2)
    gmsh.model.mesh.field.setNumber(t, 'InField', 1)
    gmsh.model.mesh.field.setNumber(t, 'SizeMin', lc_min)
    gmsh.model.mesh.field.setNumber(t, 'SizeMax', lc_max)
    gmsh.model.mesh.field.setNumber(t, 'DistMin', 0.02)
    gmsh.model.mesh.field.setNumber(t, 'DistMax', dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber('Mesh.MeshSizeMax', 0.08)

    # --- explicit boundary-layer prisms off the fin ---
    bl = gmsh.model.mesh.field.add('BoundaryLayer', 3)
    gmsh.model.mesh.field.setNumbers(bl, 'SurfacesList', groups['tailfin'])
    gmsh.model.mesh.field.setNumber(bl, 'hwall_n', hfirst)
    gmsh.model.mesh.field.setNumber(bl, 'ratio', ratio)
    gmsh.model.mesh.field.setNumber(bl, 'thickness', bl_thickness)
    gmsh.model.mesh.field.setNumber(bl, 'Quads', 1)
    gmsh.model.mesh.field.setAsBoundaryLayer(bl)

    gmsh.model.mesh.generate(3)
    os.makedirs(os.path.dirname(os.path.abspath(out_msh)), exist_ok=True)
    gmsh.write(out_msh)
    n = gmsh.model.mesh.getNumberOfElements()
    print(f'wrote {out_msh}  total elements: {n}')
    print('patch faces:', {k: len(v) for k, v in groups.items()})
    gmsh.finalize()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', type=float, required=True)
    ap.add_argument('--ar', type=float, required=True)
    ap.add_argument('--taper', type=float, required=True)
    ap.add_argument('--dat', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  '..', 'cad_generator', 'sd8020.dat'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--hfirst', type=float, default=2.5e-05)
    ap.add_argument('--ratio', type=float, default=1.25)
    ap.add_argument('--thickness', type=float, default=1.5e-03)
    ap.add_argument('--lc-min', type=float, default=0.002)
    a = ap.parse_args()
    build(a.sweep, a.ar, a.taper, a.dat, a.out,
          hfirst=a.hfirst, ratio=a.ratio, bl_thickness=a.thickness, lc_min=a.lc_min)
